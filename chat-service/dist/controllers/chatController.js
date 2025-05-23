"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.searchUsers = exports.getOnlineUsers = exports.updateConversation = exports.removeParticipant = exports.addParticipant = exports.getConversationMessages = exports.getConversations = exports.createConversation = void 0;
const app_1 = require("../app");
const uuid_1 = require("uuid");
const createConversation = async (req, res) => {
    try {
        const { participantIds, name, type = 'direct' } = req.body;
        const userId = req.userId;
        if (!participantIds || !Array.isArray(participantIds) || participantIds.length === 0) {
            return res.status(400).json({ error: 'Participant IDs are required' });
        }
        if (type === 'direct' && participantIds.length !== 1) {
            return res.status(400).json({ error: 'Direct chats must have exactly 2 participants' });
        }
        if (type === 'direct') {
            const existingConversation = await app_1.prisma.conversation.findFirst({
                where: {
                    type: 'direct',
                    participants: {
                        every: {
                            userId: { in: [userId, participantIds[0]] }
                        }
                    }
                },
                include: {
                    participants: true
                }
            });
            if (existingConversation && existingConversation.participants.length === 2) {
                return res.json({
                    message: 'Conversation already exists',
                    conversation: existingConversation
                });
            }
        }
        const conversation = await app_1.prisma.conversation.create({
            data: {
                id: (0, uuid_1.v4)(),
                name: type === 'group' ? name : null,
                type,
                createdBy: userId,
                participants: {
                    create: [
                        { userId },
                        ...participantIds.map((id) => ({ userId: id }))
                    ]
                }
            },
            include: {
                participants: {
                    select: {
                        userId: true,
                        joinedAt: true
                    }
                }
            }
        });
        res.status(201).json({
            message: 'Conversation created successfully',
            conversation
        });
    }
    catch (error) {
        console.error('Create conversation error:', error);
        res.status(500).json({ error: 'Failed to create conversation' });
    }
};
exports.createConversation = createConversation;
const getConversations = async (req, res) => {
    try {
        const userId = req.userId;
        const { page = 1, limit = 20 } = req.query;
        const skip = (Number(page) - 1) * Number(limit);
        const conversations = await app_1.prisma.conversation.findMany({
            where: {
                participants: {
                    some: {
                        userId,
                        leftAt: null
                    }
                }
            },
            include: {
                participants: {
                    where: { leftAt: null },
                    select: {
                        userId: true,
                        lastReadAt: true
                    }
                },
                messages: {
                    orderBy: { createdAt: 'desc' },
                    take: 1,
                    select: {
                        id: true,
                        content: true,
                        senderId: true,
                        messageType: true,
                        createdAt: true
                    }
                },
                _count: {
                    select: {
                        messages: true,
                        participants: {
                            where: { leftAt: null }
                        }
                    }
                }
            },
            orderBy: { updatedAt: 'desc' },
            skip,
            take: Number(limit)
        });
        const conversationsWithUnread = await Promise.all(conversations.map(async (conv) => {
            const participant = conv.participants.find(p => p.userId === userId);
            const lastReadAt = participant?.lastReadAt || new Date(0);
            const unreadCount = await app_1.prisma.message.count({
                where: {
                    conversationId: conv.id,
                    createdAt: { gt: lastReadAt },
                    senderId: { not: userId }
                }
            });
            return {
                ...conv,
                unreadCount
            };
        }));
        res.json({
            conversations: conversationsWithUnread,
            pagination: {
                page: Number(page),
                limit: Number(limit),
                hasMore: conversations.length === Number(limit)
            }
        });
    }
    catch (error) {
        console.error('Get conversations error:', error);
        res.status(500).json({ error: 'Failed to get conversations' });
    }
};
exports.getConversations = getConversations;
const getConversationMessages = async (req, res) => {
    try {
        const { conversationId } = req.params;
        const userId = req.userId;
        const { page = 1, limit = 50, before } = req.query;
        const participant = await app_1.prisma.conversationParticipant.findUnique({
            where: {
                conversationId_userId: {
                    conversationId,
                    userId
                }
            }
        });
        if (!participant) {
            return res.status(403).json({ error: 'Not authorized to view this conversation' });
        }
        const skip = (Number(page) - 1) * Number(limit);
        const whereClause = {
            conversationId,
            deletedAt: null
        };
        if (before) {
            whereClause.createdAt = { lt: new Date(before) };
        }
        const messages = await app_1.prisma.message.findMany({
            where: whereClause,
            orderBy: { createdAt: 'desc' },
            skip,
            take: Number(limit),
            include: {
                readReceipts: {
                    select: {
                        userId: true,
                        readAt: true
                    }
                }
            }
        });
        res.json({
            messages: messages.reverse(),
            pagination: {
                page: Number(page),
                limit: Number(limit),
                hasMore: messages.length === Number(limit)
            }
        });
    }
    catch (error) {
        console.error('Get messages error:', error);
        res.status(500).json({ error: 'Failed to get messages' });
    }
};
exports.getConversationMessages = getConversationMessages;
const addParticipant = async (req, res) => {
    try {
        const { conversationId } = req.params;
        const { userId: participantId } = req.body;
        const userId = req.userId;
        const conversation = await app_1.prisma.conversation.findFirst({
            where: {
                id: conversationId,
                participants: {
                    some: { userId }
                }
            }
        });
        if (!conversation) {
            return res.status(404).json({ error: 'Conversation not found or access denied' });
        }
        if (conversation.type !== 'group') {
            return res.status(400).json({ error: 'Can only add participants to group conversations' });
        }
        const existingParticipant = await app_1.prisma.conversationParticipant.findUnique({
            where: {
                conversationId_userId: {
                    conversationId,
                    userId: participantId
                }
            }
        });
        if (existingParticipant && !existingParticipant.leftAt) {
            return res.status(400).json({ error: 'User is already a participant' });
        }
        const participant = await app_1.prisma.conversationParticipant.upsert({
            where: {
                conversationId_userId: {
                    conversationId,
                    userId: participantId
                }
            },
            update: {
                leftAt: null,
                joinedAt: new Date()
            },
            create: {
                conversationId,
                userId: participantId
            }
        });
        await app_1.prisma.message.create({
            data: {
                conversationId,
                senderId: userId,
                content: `User ${participantId} was added to the conversation`,
                messageType: 'system'
            }
        });
        res.json({
            message: 'Participant added successfully',
            participant
        });
    }
    catch (error) {
        console.error('Add participant error:', error);
        res.status(500).json({ error: 'Failed to add participant' });
    }
};
exports.addParticipant = addParticipant;
const removeParticipant = async (req, res) => {
    try {
        const { conversationId, userId: participantId } = req.params;
        const userId = req.userId;
        const conversation = await app_1.prisma.conversation.findFirst({
            where: {
                id: conversationId,
                participants: {
                    some: { userId }
                }
            }
        });
        if (!conversation) {
            return res.status(404).json({ error: 'Conversation not found or access denied' });
        }
        if (conversation.type !== 'group') {
            return res.status(400).json({ error: 'Can only remove participants from group conversations' });
        }
        await app_1.prisma.conversationParticipant.update({
            where: {
                conversationId_userId: {
                    conversationId,
                    userId: participantId
                }
            },
            data: {
                leftAt: new Date()
            }
        });
        await app_1.prisma.message.create({
            data: {
                conversationId,
                senderId: userId,
                content: `User ${participantId} was removed from the conversation`,
                messageType: 'system'
            }
        });
        res.json({
            message: 'Participant removed successfully'
        });
    }
    catch (error) {
        console.error('Remove participant error:', error);
        res.status(500).json({ error: 'Failed to remove participant' });
    }
};
exports.removeParticipant = removeParticipant;
const updateConversation = async (req, res) => {
    try {
        const { conversationId } = req.params;
        const { name } = req.body;
        const userId = req.userId;
        const participant = await app_1.prisma.conversationParticipant.findUnique({
            where: {
                conversationId_userId: {
                    conversationId,
                    userId
                }
            }
        });
        if (!participant) {
            return res.status(403).json({ error: 'Not authorized to update this conversation' });
        }
        const conversation = await app_1.prisma.conversation.update({
            where: { id: conversationId },
            data: { name }
        });
        res.json({
            message: 'Conversation updated successfully',
            conversation
        });
    }
    catch (error) {
        console.error('Update conversation error:', error);
        res.status(500).json({ error: 'Failed to update conversation' });
    }
};
exports.updateConversation = updateConversation;
const getOnlineUsers = async (req, res) => {
    try {
        const onlineUsers = await app_1.prisma.userStatus.findMany({
            where: { isOnline: true },
            select: {
                userId: true,
                lastSeenAt: true
            }
        });
        res.json({
            onlineUsers,
            count: onlineUsers.length
        });
    }
    catch (error) {
        console.error('Get online users error:', error);
        res.status(500).json({ error: 'Failed to get online users' });
    }
};
exports.getOnlineUsers = getOnlineUsers;
const searchUsers = async (req, res) => {
    try {
        const { q: query, limit = 10 } = req.query;
        if (!query) {
            return res.status(400).json({ error: 'Search query is required' });
        }
        // Note: This would typically search in a Users table
        // For now, return placeholder data
        const users = [];
        res.json({
            users,
            query
        });
    }
    catch (error) {
        console.error('Search users error:', error);
        res.status(500).json({ error: 'Failed to search users' });
    }
};
exports.searchUsers = searchUsers;
