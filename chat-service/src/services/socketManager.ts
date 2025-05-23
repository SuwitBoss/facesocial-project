import { Server, Socket } from 'socket.io';
import { PrismaClient } from '@prisma/client';
import { v4 as uuidv4 } from 'uuid';

export class SocketManager {
  private io: Server;
  private prisma: PrismaClient;
  private redis: any;
  private userSockets: Map<string, string[]> = new Map(); // userId -> socketIds[]

  constructor(io: Server, prisma: PrismaClient, redis: any) {
    this.io = io;
    this.prisma = prisma;
    this.redis = redis;
  }

  async handleConnection(socket: Socket) {
    const userId = socket.userId;
    if (!this.userSockets.has(userId)) {
      this.userSockets.set(userId, []);
    }
    this.userSockets.get(userId)!.push(socket.id);
    await this.updateUserStatus(userId, true, socket.id);
    await this.joinUserConversations(socket, userId);
    this.broadcastUserStatus(userId, true);
    await this.sendConversationList(socket, userId);
  }

  async handleDisconnection(socket: Socket) {
    const userId = socket.userId;
    if (this.userSockets.has(userId)) {
      const sockets = this.userSockets.get(userId)!;
      const index = sockets.indexOf(socket.id);
      if (index > -1) {
        sockets.splice(index, 1);
      }
      if (sockets.length === 0) {
        this.userSockets.delete(userId);
        await this.updateUserStatus(userId, false);
        this.broadcastUserStatus(userId, false);
      }
    }
  }

  async handleJoinConversation(socket: Socket, data: { conversationId: string }) {
    const { conversationId } = data;
    const participant = await this.prisma.conversationParticipant.findUnique({
      where: {
        conversationId_userId: {
          conversationId,
          userId: socket.userId
        }
      }
    });
    if (participant) {
      socket.join(conversationId);
      await this.prisma.conversationParticipant.update({
        where: { id: participant.id },
        data: { lastReadAt: new Date() }
      });
      socket.emit('joined_conversation', { conversationId });
      await this.sendRecentMessages(socket, conversationId);
    } else {
      socket.emit('error', { message: 'Not authorized to join this conversation' });
    }
  }

  async handleLeaveConversation(socket: Socket, data: { conversationId: string }) {
    const { conversationId } = data;
    socket.leave(conversationId);
    socket.emit('left_conversation', { conversationId });
  }

  async handleSendMessage(socket: Socket, data: { 
    conversationId: string; 
    content: string; 
    messageType?: string;
    attachmentUrl?: string;
    attachmentType?: string;
    attachmentSize?: number;
  }) {
    try {
      const { conversationId, content, messageType = 'text', attachmentUrl, attachmentType, attachmentSize } = data;
      const participant = await this.prisma.conversationParticipant.findUnique({
        where: {
          conversationId_userId: {
            conversationId,
            userId: socket.userId
          }
        }
      });
      if (!participant) {
        socket.emit('error', { message: 'Not authorized to send message to this conversation' });
        return;
      }
      const message = await this.prisma.message.create({
        data: {
          id: uuidv4(),
          conversationId,
          senderId: socket.userId,
          content,
          messageType,
          attachmentUrl,
          attachmentType,
          attachmentSize
        }
      });
      await this.prisma.conversation.update({
        where: { id: conversationId },
        data: { updatedAt: new Date() }
      });
      this.io.to(conversationId).emit('new_message', {
        id: message.id,
        conversationId,
        senderId: socket.userId,
        content,
        messageType,
        attachmentUrl,
        attachmentType,
        attachmentSize,
        createdAt: message.createdAt
      });
      await this.sendPushNotifications(conversationId, socket.userId, content);
    } catch (error) {
      console.error('Send message error:', error);
      socket.emit('error', { message: 'Failed to send message' });
    }
  }

  async handleTypingStart(socket: Socket, data: { conversationId: string }) {
    const { conversationId } = data;
    socket.to(conversationId).emit('user_typing', {
      userId: socket.userId,
      conversationId,
      isTyping: true
    });
  }

  async handleTypingStop(socket: Socket, data: { conversationId: string }) {
    const { conversationId } = data;
    socket.to(conversationId).emit('user_typing', {
      userId: socket.userId,
      conversationId,
      isTyping: false
    });
  }

  async handleMarkRead(socket: Socket, data: { messageId: string }) {
    try {
      const { messageId } = data;
      await this.prisma.messageRead.upsert({
        where: {
          messageId_userId: {
            messageId,
            userId: socket.userId
          }
        },
        update: {
          readAt: new Date()
        },
        create: {
          messageId,
          userId: socket.userId,
          readAt: new Date()
        }
      });
      const message = await this.prisma.message.findUnique({
        where: { id: messageId }
      });
      if (message) {
        socket.to(message.conversationId).emit('message_read', {
          messageId,
          userId: socket.userId,
          readAt: new Date()
        });
      }
    } catch (error) {
      console.error('Mark read error:', error);
    }
  }

  async handleGetOnlineUsers(socket: Socket) {
    const onlineUsers = Array.from(this.userSockets.keys());
    socket.emit('online_users', { users: onlineUsers });
  }

  private async updateUserStatus(userId: string, isOnline: boolean, socketId?: string) {
    await this.prisma.userStatus.upsert({
      where: { userId },
      update: {
        isOnline,
        lastSeenAt: new Date(),
        socketId: isOnline ? socketId : null
      },
      create: {
        userId,
        isOnline,
        lastSeenAt: new Date(),
        socketId: isOnline ? socketId : null
      }
    });
  }

  private broadcastUserStatus(userId: string, isOnline: boolean) {
    this.io.emit('user_status_changed', {
      userId,
      isOnline,
      timestamp: new Date()
    });
  }

  private async joinUserConversations(socket: Socket, userId: string) {
    const participants = await this.prisma.conversationParticipant.findMany({
      where: {
        userId,
        leftAt: null
      },
      select: {
        conversationId: true
      }
    });
    for (const participant of participants) {
      socket.join(participant.conversationId);
    }
  }

  private async sendConversationList(socket: Socket, userId: string) {
    const conversations = await this.prisma.conversation.findMany({
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
            messages: {
              where: {
                createdAt: {
                  gt: await this.getUserLastReadTime(userId)
                }
              }
            }
          }
        }
      },
      orderBy: { updatedAt: 'desc' }
    });
    socket.emit('conversation_list', { conversations });
  }

  private async sendRecentMessages(socket: Socket, conversationId: string) {
    const messages = await this.prisma.message.findMany({
      where: {
        conversationId,
        deletedAt: null
      },
      orderBy: { createdAt: 'desc' },
      take: 50,
      include: {
        readReceipts: {
          select: {
            userId: true,
            readAt: true
          }
        }
      }
    });
    socket.emit('recent_messages', {
      conversationId,
      messages: messages.reverse()
    });
  }

  private async getUserLastReadTime(userId: string): Promise<Date> {
    const participant = await this.prisma.conversationParticipant.findFirst({
      where: { userId },
      orderBy: { lastReadAt: 'desc' },
      select: { lastReadAt: true }
    });
    return participant?.lastReadAt || new Date(0);
  }

  private async sendPushNotifications(conversationId: string, senderId: string, content: string) {
    const participants = await this.prisma.conversationParticipant.findMany({
      where: {
        conversationId,
        userId: { not: senderId },
        leftAt: null
      },
      include: {
        conversation: {
          select: { name: true, type: true }
        }
      }
    });
    // TODO: Implement actual push notification service
    console.log(`📱 Would send push notification to ${participants.length} users`);
  }
}
