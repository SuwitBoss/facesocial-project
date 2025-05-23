import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import cors from 'cors';
import dotenv from 'dotenv';
import { PrismaClient } from '@prisma/client';
import { createClient } from 'redis';
import jwt from 'jsonwebtoken';
import chatRoutes from './routes/chatRoutes';
import { SocketManager } from './services/socketManager';

dotenv.config();

const app = express();
const server = createServer(app);
const prisma = new PrismaClient();
const PORT = process.env.PORT || 3005;

// Redis client
export const redisClient = createClient({
  url: process.env.REDIS_URL || 'redis://redis:6379'
});

// Socket.IO setup
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/chat', chatRoutes);

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'OK',
    service: 'Chat Service',
    timestamp: new Date().toISOString(),
    connections: io.engine.clientsCount
  });
});

// Socket.IO Authentication Middleware
io.use((socket, next) => {
  try {
    const token = socket.handshake.auth.token || socket.handshake.headers.authorization?.replace('Bearer ', '');
    if (!token) {
      return next(new Error('Authentication error: No token provided'));
    }
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'fallback-secret') as any;
    socket.userId = decoded.userId;
    next();
  } catch (err) {
    next(new Error('Authentication error: Invalid token'));
  }
});

// Initialize Socket Manager
const socketManager = new SocketManager(io, prisma, redisClient);

// Socket.IO Connection Handler
io.on('connection', (socket) => {
  console.log(`🔌 User ${socket.userId} connected with socket ${socket.id}`);
  socketManager.handleConnection(socket);
  socket.on('disconnect', () => {
    console.log(`🔌 User ${socket.userId} disconnected`);
    socketManager.handleDisconnection(socket);
  });
  socket.on('join_conversation', (data) => {
    socketManager.handleJoinConversation(socket, data);
  });
  socket.on('leave_conversation', (data) => {
    socketManager.handleLeaveConversation(socket, data);
  });
  socket.on('send_message', (data) => {
    socketManager.handleSendMessage(socket, data);
  });
  socket.on('typing_start', (data) => {
    socketManager.handleTypingStart(socket, data);
  });
  socket.on('typing_stop', (data) => {
    socketManager.handleTypingStop(socket, data);
  });
  socket.on('mark_read', (data) => {
    socketManager.handleMarkRead(socket, data);
  });
  socket.on('get_online_users', () => {
    socketManager.handleGetOnlineUsers(socket);
  });
});

redisClient.on('error', err => console.log('Redis Client Error', err));
redisClient.connect().then(() => {
  console.log('📦 Connected to Redis');
});

server.listen(PORT, () => {
  console.log(`🚀 Chat Service running on port ${PORT}`);
  console.log(`💬 Socket.IO ready for real-time messaging`);
  console.log(`📡 Health check: http://localhost:${PORT}/health`);
});

const shutdown = async () => {
  try {
    await prisma.$disconnect();
    await redisClient.quit();
    io.close();
    console.log('🔌 Chat Service shutdown complete.');
  } catch (err) {
    console.error('Error during shutdown:', err);
  } finally {
    process.exit(0);
  }
};

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

export { prisma, io };

declare module 'socket.io' {
  interface Socket {
    userId: string;
  }
}
