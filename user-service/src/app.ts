import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { PrismaClient } from '@prisma/client';
import userRoutes from './routes/userRoutes';

// Load environment variables
dotenv.config();

const app = express();
const prisma = new PrismaClient();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/users', userRoutes);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    service: 'User Service',
    timestamp: new Date().toISOString(),
    endpoints: [
      'POST /api/users/register',
      'POST /api/users/login', 
      'GET /api/users/profile'
    ]
  });
});

// Test database connection
app.get('/test-db', async (req, res) => {
  try {
    await prisma.$connect();
    res.json({ 
      status: 'Database connected successfully',
      database: 'PostgreSQL'
    });
  } catch (error) {
    res.status(500).json({ 
      status: 'Database connection failed',
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 User Service running on port ${PORT}`);
  console.log(`📡 Health check: http://localhost:${PORT}/health`);
  console.log(`🗄️  Database test: http://localhost:${PORT}/test-db`);
  console.log(`👤 User API endpoints:`);
  console.log(`   POST http://localhost:${PORT}/api/users/register`);
  console.log(`   POST http://localhost:${PORT}/api/users/login`);
  console.log(`   GET  http://localhost:${PORT}/api/users/profile`);
});

// Graceful shutdown
process.on('beforeExit', async () => {
  await prisma.$disconnect();
});
