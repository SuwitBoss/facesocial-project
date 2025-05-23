import axios from 'axios';
import FormData from 'form-data';

const FACE_SERVICE_URL = process.env.FACE_SERVICE_URL || 'http://face-recognition-service:3002';
const DEEPFAKE_SERVICE_URL = process.env.DEEPFAKE_SERVICE_URL || 'http://deepfake-detection-service:8000';

export interface FaceDetectionResult {
  faces_count: number;
  faces: Array<{
    bbox: number[];
    confidence: number;
  }>;
}

export interface FaceIdentificationResult {
  identified: boolean;
  best_match?: {
    user_id: string;
    similarity: number;
  };
  candidates: Array<{
    user_id: string;
    similarity: number;
  }>;
}

export interface DeepfakeResult {
  is_deepfake: boolean;
  deepfake_probability: number;
  confidence: number;
}

export class FaceService {
  async detectFaces(imageBuffer: Buffer): Promise<FaceDetectionResult> {
    const formData = new FormData();
    formData.append('file', imageBuffer, { filename: 'image.jpg' });
    formData.append('confidence_threshold', '0.7');

    const response = await axios.post(
      `${FACE_SERVICE_URL}/face/detect`,
      formData,
      {
        headers: formData.getHeaders()
      }
    );

    return response.data;
  }

  async identifyFaces(imageBuffer: Buffer, threshold = 0.7): Promise<FaceIdentificationResult> {
    const formData = new FormData();
    formData.append('file', imageBuffer, { filename: 'image.jpg' });
    formData.append('top_k', '10');
    formData.append('threshold', threshold.toString());

    const response = await axios.post(
      `${FACE_SERVICE_URL}/face/identify`,
      formData,
      {
        headers: formData.getHeaders()
      }
    );

    return response.data;
  }

  async checkDeepfake(imageBuffer: Buffer): Promise<DeepfakeResult> {
    const formData = new FormData();
    formData.append('file', imageBuffer, { filename: 'image.jpg' });
    formData.append('threshold', '0.31');

    const response = await axios.post(
      `${DEEPFAKE_SERVICE_URL}/deepfake/detect`,
      formData,
      {
        headers: formData.getHeaders()
      }
    );

    return response.data;
  }

  // Helper to extract face from image based on bbox
  async extractFaceImage(imageBuffer: Buffer, bbox: number[]): Promise<Buffer> {
    // This would use sharp or jimp to crop the face
    // For now, return the original buffer
    return imageBuffer;
  }
}

export const faceService = new FaceService();
