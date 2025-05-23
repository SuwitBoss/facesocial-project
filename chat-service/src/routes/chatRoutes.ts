import { Router } from 'express';
import { 
  createConversation, 
  getConversations, 
  getConversationMessages,
  addParticipant,
  removeParticipant,
  updateConversation,
  getOnlineUsers,
  searchUsers
} from '../controllers/chatController';
import { authMiddleware } from '../middleware/auth';

const router = Router();

// All routes require authentication
router.use(authMiddleware);

// Conversation routes
router.post('/conversations', createConversation);
router.get('/conversations', getConversations);
router.get('/conversations/:conversationId/messages', getConversationMessages);
router.put('/conversations/:conversationId', updateConversation);

// Participants routes
router.post('/conversations/:conversationId/participants', addParticipant);
router.delete('/conversations/:conversationId/participants/:userId', removeParticipant);

// User routes
router.get('/users/online', getOnlineUsers);
router.get('/users/search', searchUsers);

export default router;
