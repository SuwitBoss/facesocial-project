"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = require("express");
const chatController_1 = require("../controllers/chatController");
const auth_1 = require("../middleware/auth");
const router = (0, express_1.Router)();
// All routes require authentication
router.use(auth_1.authMiddleware);
// Conversation routes
router.post('/conversations', chatController_1.createConversation);
router.get('/conversations', chatController_1.getConversations);
router.get('/conversations/:conversationId/messages', chatController_1.getConversationMessages);
router.put('/conversations/:conversationId', chatController_1.updateConversation);
// Participants routes
router.post('/conversations/:conversationId/participants', chatController_1.addParticipant);
router.delete('/conversations/:conversationId/participants/:userId', chatController_1.removeParticipant);
// User routes
router.get('/users/online', chatController_1.getOnlineUsers);
router.get('/users/search', chatController_1.searchUsers);
exports.default = router;
