import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { PrismaClient } from '@prisma/client';
import postRoutes from './routes/postRoutes';
import { createClient } from 'redis';
import multer from 'multer';

dotenv.config();

const app = express();
const prisma = new PrismaClient();
const PORT = process.env.PORT || 3004;

// Redis client
export const redisClient = createClient({
  url: process.env.REDIS_URL || 'redis://redis:6379'
});

// Multer for file uploads
export const upload = multer({
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

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/posts', postRoutes);

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'OK',
    service: 'Post Service',
    timestamp: new Date().toISOString()
  });
});

// Initialize Redis
redisClient.on('error', err => console.log('Redis Client Error', err));
redisClient.connect().then(() => {
  console.log('📦 Connected to Redis');
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Post Service running on port ${PORT}`);
});

// Graceful shutdown
const shutdown = async () => {
  try {
    await prisma.$disconnect();
    await redisClient.quit();
    console.log('🔌 Graceful shutdown complete.');
  } catch (err) {
    console.error('Error during shutdown:', err);
  } finally {
    process.exit(0);
  }
};

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
process.on('beforeExit', async () => {
  await prisma.$disconnect();
  await redisClient.quit();
});

export { prisma };
