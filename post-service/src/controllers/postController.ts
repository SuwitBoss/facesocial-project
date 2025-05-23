import { Request, Response } from 'express';
import { prisma } from '../app';
import { faceService } from '../services/faceService';
import { v4 as uuidv4 } from 'uuid';

export interface AuthRequest extends Request {
  userId?: string;
}

export const createPost = async (req: AuthRequest, res: Response) => {
  try {
    const { content, visibility = 'public', autoTag = true } = req.body;
    const userId = req.userId!;
    const files = req.files as Express.Multer.File[];

    if (!content && (!files || files.length === 0)) {
      return res.status(400).json({ error: 'Post must have content or images' });
    }

    // Create post
    const post = await prisma.post.create({
      data: {
        userId,
        content,
        visibility
      }
    });

    // Process images if any
    const processedImages = [];
    
    for (let i = 0; i < (files?.length || 0); i++) {
      const file = files[i];
      
      // Upload to storage (simplified - in production use MinIO)
      const imageUrl = `/uploads/${uuidv4()}.jpg`; // Placeholder
      
      // Create image record
      const postImage = await prisma.postImage.create({
        data: {
          postId: post.id,
          imageUrl,
          mimeType: file.mimetype,
          size: file.size,
          order: i
        }
      });

      // Auto-tag faces if enabled
      if (autoTag) {
        try {
          // Detect faces
          const detection = await faceService.detectFaces(file.buffer);
          
          // Identify each face
          for (const face of detection.faces) {
            const faceBuffer = await faceService.extractFaceImage(file.buffer, face.bbox);
            const identification = await faceService.identifyFaces(faceBuffer);
            
            if (identification.best_match && identification.best_match.similarity > 0.8) {
              // Create auto face tag
              await prisma.faceTag.create({
                data: {
                  imageId: postImage.id,
                  userId: identification.best_match.user_id,
                  taggerId: userId,
                  x: face.bbox[0],
                  y: face.bbox[1],
                  width: face.bbox[2] - face.bbox[0],
                  height: face.bbox[3] - face.bbox[1],
                  confidence: identification.best_match.similarity,
                  isAutoTag: true,
                  isVerified: false
                }
              });
            }
          }
        } catch (error) {
          console.error('Auto-tagging failed:', error);
          // Continue without auto-tags
        }
      }

      processedImages.push(postImage);
    }

    // Get complete post with relations
    const completePost = await prisma.post.findUnique({
      where: { id: post.id },
      include: {
        images: {
          include: {
            faceTags: true
          }
        },
        likes: true,
        comments: {
          take: 3,
          orderBy: { createdAt: 'desc' }
        },
        _count: {
          select: {
            likes: true,
            comments: true,
            shares: true
          }
        }
      }
    });

    res.status(201).json({
      message: 'Post created successfully',
      post: completePost
    });

  } catch (error) {
    console.error('Create post error:', error);
    res.status(500).json({ error: 'Failed to create post' });
  }
};

export const getFeed = async (req: AuthRequest, res: Response) => {
  try {
    const userId = req.userId!;
    const { page = 1, limit = 10 } = req.query;
    
    const skip = (Number(page) - 1) * Number(limit);

    // For now, get all public posts (later: add following logic)
    const posts = await prisma.post.findMany({
      where: {
        visibility: 'public'
      },
      include: {
        images: {
          include: {
            faceTags: true
          }
        },
        likes: true,
        comments: {
          take: 3,
          orderBy: { createdAt: 'desc' }
        },
        _count: {
          select: {
            likes: true,
            comments: true,
            shares: true
          }
        }
      },
      orderBy: { createdAt: 'desc' },
      skip,
      take: Number(limit)
    });

    // Check if current user liked each post
    const postsWithUserLike = posts.map(post => ({
      ...post,
      isLikedByUser: post.likes.some(like => like.userId === userId)
    }));

    res.json({
      posts: postsWithUserLike,
      pagination: {
        page: Number(page),
        limit: Number(limit),
        hasMore: posts.length === Number(limit)
      }
    });

  } catch (error) {
    console.error('Get feed error:', error);
    res.status(500).json({ error: 'Failed to get feed' });
  }
};

export const getTaggedPosts = async (req: AuthRequest, res: Response) => {
  try {
    const { userId } = req.params;
    const { page = 1, limit = 10 } = req.query;
    
    const skip = (Number(page) - 1) * Number(limit);

    // Find posts where user is tagged
    const posts = await prisma.post.findMany({
      where: {
        images: {
          some: {
            faceTags: {
              some: {
                userId: userId
              }
            }
          }
        }
      },
      include: {
        images: {
          include: {
            faceTags: {
              where: {
                userId: userId
              }
            }
          }
        },
        _count: {
          select: {
            likes: true,
            comments: true,
            shares: true
          }
        }
      },
      orderBy: { createdAt: 'desc' },
      skip,
      take: Number(limit)
    });

    res.json({
      posts,
      pagination: {
        page: Number(page),
        limit: Number(limit),
        hasMore: posts.length === Number(limit)
      }
    });

  } catch (error) {
    console.error('Get tagged posts error:', error);
    res.status(500).json({ error: 'Failed to get tagged posts' });
  }
};

export const likePost = async (req: AuthRequest, res: Response) => {
  try {
    const { postId } = req.params;
    const userId = req.userId!;

    // Check if already liked
    const existingLike = await prisma.like.findUnique({
      where: {
        postId_userId: {
          postId,
          userId
        }
      }
    });

    if (existingLike) {
      // Unlike
      await prisma.like.delete({
        where: { id: existingLike.id }
      });
      res.json({ liked: false, message: 'Post unliked' });
    } else {
      // Like
      await prisma.like.create({
        data: {
          postId,
          userId
        }
      });
      res.json({ liked: true, message: 'Post liked' });
    }

  } catch (error) {
    console.error('Like post error:', error);
    res.status(500).json({ error: 'Failed to like/unlike post' });
  }
};

export const addComment = async (req: AuthRequest, res: Response) => {
  try {
    const { postId } = req.params;
    const { content } = req.body;
    const userId = req.userId!;

    if (!content) {
      return res.status(400).json({ error: 'Comment content is required' });
    }

    const comment = await prisma.comment.create({
      data: {
        postId,
        userId,
        content
      }
    });

    res.status(201).json({
      message: 'Comment added successfully',
      comment
    });

  } catch (error) {
    console.error('Add comment error:', error);
    res.status(500).json({ error: 'Failed to add comment' });
  }
};
