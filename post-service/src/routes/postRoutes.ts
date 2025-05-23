import { Router } from 'express';
import { createPost, getFeed, getTaggedPosts, likePost, addComment } from '../controllers/postController';
import { authMiddleware } from '../middleware/auth';
import multer from 'multer';

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
    files: 10 // Max 10 files per request
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'));
    }
  }
});

const router = Router();

// All routes require authentication
router.use(authMiddleware);

// Post routes
router.post('/', upload.array('images', 10), createPost);
router.get('/feed', getFeed);
router.get('/tagged/:userId', getTaggedPosts);
router.post('/:postId/like', likePost);
router.post('/:postId/comment', addComment);

export default router;
