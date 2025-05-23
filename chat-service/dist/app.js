"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.io = exports.prisma = exports.redisClient = void 0;
const express_1 = __importDefault(require("express"));
const http_1 = require("http");
const socket_io_1 = require("socket.io");
const cors_1 = __importDefault(require("cors"));
const dotenv_1 = __importDefault(require("dotenv"));
const client_1 = require("@prisma/client");
const redis_1 = require("redis");
const jsonwebtoken_1 = __importDefault(require("jsonwebtoken"));
const chatRoutes_1 = __importDefault(require("./routes/chatRoutes"));
const socketManager_1 = require("./services/socketManager");
dotenv_1.default.config();
const app = (0, express_1.default)();
const server = (0, http_1.createServer)(app);
const prisma = new client_1.PrismaClient();
exports.prisma = prisma;
const PORT = process.env.PORT || 3005;
// Redis client
exports.redisClient = (0, redis_1.createClient)({
    url: process.env.REDIS_URL || 'redis://redis:6379'
});
// Socket.IO setup
const io = new socket_io_1.Server(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});
exports.io = io;
// Middleware
app.use((0, cors_1.default)());
app.use(express_1.default.json());
// Routes
app.use('/api/chat', chatRoutes_1.default);
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
        const decoded = jsonwebtoken_1.default.verify(token, process.env.JWT_SECRET || 'fallback-secret');
        socket.userId = decoded.userId;
        next();
    }
    catch (err) {
        next(new Error('Authentication error: Invalid token'));
    }
});
// Initialize Socket Manager
const socketManager = new socketManager_1.SocketManager(io, prisma, exports.redisClient);
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
exports.redisClient.on('error', err => console.log('Redis Client Error', err));
exports.redisClient.connect().then(() => {
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
        await exports.redisClient.quit();
        io.close();
        console.log('🔌 Chat Service shutdown complete.');
    }
    catch (err) {
        console.error('Error during shutdown:', err);
    }
    finally {
        process.exit(0);
    }
};
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
