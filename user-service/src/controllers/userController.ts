import { Request, Response } from 'express';
import { PrismaClient } from '@prisma/client';
import { hashPassword, comparePassword, generateToken } from '../utils/auth';
import { AuthRequest } from '../middleware/auth';

const prisma = new PrismaClient();

export const register = async (req: Request, res: Response) => {
  try {
    const { email, username, password, fullName, bio } = req.body;

    // Validate required fields
    if (!email || !username || !password || !fullName) {
      res.status(400).json({ 
        error: 'Email, username, password, and fullName are required' 
      });
      return;
    }

    // Check if user already exists
    const existingUser = await prisma.user.findFirst({
      where: {
        OR: [
          { email },
          { username }
        ]
      }
    });

    if (existingUser) {
      res.status(409).json({ 
        error: 'User with this email or username already exists' 
      });
      return;
    }

    // Hash password and create user
    const passwordHash = await hashPassword(password);
    
    const newUser = await prisma.user.create({
      data: {
        email,
        username,
        passwordHash,
        fullName,
        bio: bio || null
      }
    });

    // Remove password hash from response
    const { passwordHash: _, ...userWithoutPassword } = newUser;

    res.status(201).json({
      message: 'User registered successfully',
      user: userWithoutPassword
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

export const login = async (req: Request, res: Response) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      res.status(400).json({ error: 'Email and password are required' });
      return;
    }

    // Find user by email
    const user = await prisma.user.findUnique({
      where: { email }
    });

    if (!user) {
      res.status(401).json({ error: 'Invalid credentials' });
      return;
    }

    // Check password
    const isPasswordValid = await comparePassword(password, user.passwordHash);
    
    if (!isPasswordValid) {
      res.status(401).json({ error: 'Invalid credentials' });
      return;
    }

    // Generate token
    const token = generateToken(user.id);
    
    // Remove password hash from response
    const { passwordHash: _, ...userWithoutPassword } = user;

    res.json({
      message: 'Login successful',
      token,
      user: userWithoutPassword
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

export const getProfile = async (req: AuthRequest, res: Response) => {
  try {
    const userId = req.userId;

    const user = await prisma.user.findUnique({
      where: { id: userId }
    });

    if (!user) {
      res.status(404).json({ error: 'User not found' });
      return;
    }

    // Remove password hash from response
    const { passwordHash: _, ...userWithoutPassword } = user;

    res.json({
      user: userWithoutPassword
    });
  } catch (error) {
    console.error('Get profile error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};
