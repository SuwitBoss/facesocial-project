#!/usr/bin/env python3
"""
FaceSocial Complete System Test
ระบบทดสอบทั้งหมดสำหรับโครงการ FaceSocial AI Services

Test Coverage:
1. Face Recognition Service
2. Anti-Spoofing Service  
3. Deepfake Detection Service
4. Face Detection Services (MTCNN, YOLO, MediaPipe)
5. Gender & Age Detection Service
6. API Gateway Integration
7. End-to-End Pipeline Testing

Author: FaceSocial Development Team
Date: May 31, 2025
"""

import os
import sys
import asyncio
import aiohttp
import json
import base64
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_test_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FaceSocialSystemTester:
    """Complete system tester for FaceSocial AI Services"""
    
    def __init__(self):
        self.base_url = "http://localhost"
        self.services = {
            'face_recognition': 5001,
            'antispoof': 5002,
            'deepfake': 5003,
            'mtcnn': 5004,
            'gender_age': 5005,
            'mediapipe': 5006,
            'yolo': 5007,
            'api_gateway': 8080
        }
        
        # Test images directory
        self.test_images_dir = Path("test-image")
        
        # Test results storage
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'service_status': {},
            'performance_metrics': {},
            'detailed_results': []
        }
        
        # Session for HTTP requests
        self.session = None
    
    async def setup_session(self):
        """Initialize HTTP session"""
        connector = aiohttp.TCPConnector(limit=100, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
    
    async def cleanup_session(self):
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        try:
            with open(image_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return ""
    
    def get_test_images(self) -> Dict[str, List[str]]:
        """Get categorized test images"""
        images = {
            'real': [],
            'fake': [],
            'spoof': [],
            'group': [],
            'test': []
        }
        
        if not self.test_images_dir.exists():
            logger.warning(f"Test images directory not found: {self.test_images_dir}")
            return images
        
        for img_file in self.test_images_dir.glob("*.jpg"):
            img_name = img_file.name.lower()
            if 'real' in img_name:
                images['real'].append(str(img_file))
            elif 'fake' in img_name:
                images['fake'].append(str(img_file))
            elif 'spoof' in img_name:
                images['spoof'].append(str(img_file))
            elif 'group' in img_name:
                images['group'].append(str(img_file))
            elif 'test' in img_name:
                images['test'].append(str(img_file))
        
        return images
    
    async def test_service_health(self, service_name: str, port: int) -> bool:
        """Test if service is running and healthy"""
        try:
            url = f"{self.base_url}:{port}/health"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    self.test_results['service_status'][service_name] = {
                        'status': 'healthy',
                        'response_time': response.headers.get('X-Response-Time', 'N/A'),
                        'data': data
                    }
                    logger.info(f"✅ {service_name} service is healthy")
                    return True
                else:
                    self.test_results['service_status'][service_name] = {
                        'status': 'unhealthy',
                        'status_code': response.status
                    }
                    logger.error(f"❌ {service_name} service unhealthy: {response.status}")
                    return False
        except Exception as e:
            self.test_results['service_status'][service_name] = {
                'status': 'error',
                'error': str(e)
            }
            logger.error(f"❌ {service_name} service error: {e}")
            return False
    
    async def test_face_recognition_service(self) -> Dict[str, Any]:
        """Test Face Recognition Service"""
        logger.info("🔍 Testing Face Recognition Service...")
        results = {'registration': [], 'recognition': [], 'comparison': []}
        
        test_images = self.get_test_images()
        port = self.services['face_recognition']
        
        # Test 1: Face Registration
        for i, img_path in enumerate(test_images['real'][:3]):
            try:
                start_time = time.time()
                image_data = self.encode_image_to_base64(img_path)
                
                payload = {
                    'user_id': f'test_user_{i}',
                    'image': image_data,
                    'name': f'Test User {i}',
                    'metadata': {'test': True}
                }
                
                url = f"{self.base_url}:{port}/register"
                async with self.session.post(url, json=payload) as response:
                    response_time = time.time() - start_time
                    data = await response.json()
                    
                    success = response.status == 200 and data.get('success', False)
                    results['registration'].append({
                        'user_id': payload['user_id'],
                        'success': success,
                        'response_time': response_time,
                        'message': data.get('message', ''),
                        'confidence': data.get('confidence')
                    })
                    
                    if success:
                        logger.info(f"✅ Registration successful for {payload['user_id']}")
                    else:
                        logger.error(f"❌ Registration failed for {payload['user_id']}: {data}")
                        
            except Exception as e:
                logger.error(f"❌ Registration error for user {i}: {e}")
                results['registration'].append({
                    'user_id': f'test_user_{i}',
                    'success': False,
                    'error': str(e)
                })
        
        # Test 2: Face Recognition
        for i, img_path in enumerate(test_images['test'][:3]):
            try:
                start_time = time.time()
                image_data = self.encode_image_to_base64(img_path)
                
                payload = {'image': image_data}
                
                url = f"{self.base_url}:{port}/recognize"
                async with self.session.post(url, json=payload) as response:
                    response_time = time.time() - start_time
                    data = await response.json()
                    
                    success = response.status == 200
                    results['recognition'].append({
                        'test_image': Path(img_path).name,
                        'success': success,
                        'response_time': response_time,
                        'matches': data.get('matches', []),
                        'confidence': data.get('confidence')
                    })
                    
                    if success:
                        logger.info(f"✅ Recognition completed for {Path(img_path).name}")
                    else:
                        logger.error(f"❌ Recognition failed for {Path(img_path).name}")
                        
            except Exception as e:
                logger.error(f"❌ Recognition error for {img_path}: {e}")
                results['recognition'].append({
                    'test_image': Path(img_path).name,
                    'success': False,
                    'error': str(e)
                })
        
        # Test 3: Face Comparison
        if len(test_images['real']) >= 2:
            try:
                start_time = time.time()
                image1_data = self.encode_image_to_base64(test_images['real'][0])
                image2_data = self.encode_image_to_base64(test_images['real'][1])
                
                payload = {
                    'image1': image1_data,
                    'image2': image2_data
                }
                
                url = f"{self.base_url}:{port}/compare"
                async with self.session.post(url, json=payload) as response:
                    response_time = time.time() - start_time
                    data = await response.json()
                    
                    success = response.status == 200
                    results['comparison'].append({
                        'success': success,
                        'response_time': response_time,
                        'similarity': data.get('similarity'),
                        'is_same_person': data.get('is_same_person')
                    })
                    
                    if success:
                        logger.info(f"✅ Face comparison completed")
                    else:
                        logger.error(f"❌ Face comparison failed")
                        
            except Exception as e:
                logger.error(f"❌ Face comparison error: {e}")
                results['comparison'].append({
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    async def test_antispoof_service(self) -> Dict[str, Any]:
        """Test Anti-Spoofing Service"""
        logger.info("🛡️ Testing Anti-Spoofing Service...")
        results = {'real_tests': [], 'spoof_tests': []}
        
        test_images = self.get_test_images()
        port = self.services['antispoof']
        
        # Test real images
        for img_path in test_images['real'][:3]:
            try:
                start_time = time.time()
                image_data = self.encode_image_to_base64(img_path)
                
                payload = {'image': image_data}
                
                url = f"{self.base_url}:{port}/detect_spoof"
                async with self.session.post(url, json=payload) as response:
                    response_time = time.time() - start_time
                    data = await response.json()
                    
                    success = response.status == 200
                    is_real = data.get('is_real', False)
                    confidence = data.get('confidence', 0)
                    
                    results['real_tests'].append({
                        'image': Path(img_path).name,
                        'success': success,
                        'response_time': response_time,
                        'is_real': is_real,
                        'confidence': confidence,
                        'correct_prediction': is_real  # Should be True for real images
                    })
                    
                    if success and is_real:
                        logger.info(f"✅ Real image correctly identified: {Path(img_path).name}")
                    else:
                        logger.warning(f"⚠️ Real image misclassified: {Path(img_path).name}")
                        
            except Exception as e:
                logger.error(f"❌ Anti-spoof test error for {img_path}: {e}")
                results['real_tests'].append({
                    'image': Path(img_path).name,
                    'success': False,
                    'error': str(e)
                })
        
        # Test spoof images
        for img_path in test_images['spoof'][:3]:
            try:
                start_time = time.time()
                image_data = self.encode_image_to_base64(img_path)
                
                payload = {'image': image_data}
                
                url = f"{self.base_url}:{port}/detect_spoof"
                async with self.session.post(url, json=payload) as response:
                    response_time = time.time() - start_time
                    data = await response.json()
                    
                    success = response.status == 200
                    is_real = data.get('is_real', True)
                    confidence = data.get('confidence', 0)
                    
                    results['spoof_tests'].append({
                        'image': Path(img_path).name,
                        'success': success,
                        'response_time': response_time,
                        'is_real': is_real,
                        'confidence': confidence,
                        'correct_prediction': not is_real  # Should be False for spoof images
                    })
                    
                    if success and not is_real:
                        logger.info(f"✅ Spoof image correctly identified: {Path(img_path).name}")
                    else:
                        logger.warning(f"⚠️ Spoof image misclassified: {Path(img_path).name}")
                        
            except Exception as e:
                logger.error(f"❌ Anti-spoof test error for {img_path}: {e}")
                results['spoof_tests'].append({
                    'image': Path(img_path).name,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    async def test_deepfake_service(self) -> Dict[str, Any]:
        """Test Deepfake Detection Service"""
        logger.info("🎭 Testing Deepfake Detection Service...")
        results = {'real_tests': [], 'fake_tests': []}
        
        test_images = self.get_test_images()
        port = self.services['deepfake']
        
        # Test real images
        for img_path in test_images['real'][:2]:
            try:
                start_time = time.time()
                image_data = self.encode_image_to_base64(img_path)
                
                payload = {'image': image_data}
                
                url = f"{self.base_url}:{port}/detect_deepfake"
                async with self.session.post(url, json=payload) as response:
                    response_time = time.time() - start_time
                    data = await response.json()
                    
                    success = response.status == 200
                    is_fake = data.get('is_fake', True)
                    confidence = data.get('confidence', 0)
                    
                    results['real_tests'].append({
                        'image': Path(img_path).name,
                        'success': success,
                        'response_time': response_time,
                        'is_fake': is_fake,
                        'confidence': confidence,
                        'correct_prediction': not is_fake  # Should be False for real images
                    })
                    
                    if success:
                        logger.info(f"✅ Deepfake detection completed for real image: {Path(img_path).name}")
                        
            except Exception as e:
                logger.error(f"❌ Deepfake test error for {img_path}: {e}")
                results['real_tests'].append({
                    'image': Path(img_path).name,
                    'success': False,
                    'error': str(e)
                })
        
        # Test fake images
        for img_path in test_images['fake'][:2]:
            try:
                start_time = time.time()
                image_data = self.encode_image_to_base64(img_path)
                
                payload = {'image': image_data}
                
                url = f"{self.base_url}:{port}/detect_deepfake"
                async with self.session.post(url, json=payload) as response:
                    response_time = time.time() - start_time
                    data = await response.json()
                    
                    success = response.status == 200
                    is_fake = data.get('is_fake', False)
                    confidence = data.get('confidence', 0)
                    
                    results['fake_tests'].append({
                        'image': Path(img_path).name,
                        'success': success,
                        'response_time': response_time,
                        'is_fake': is_fake,
                        'confidence': confidence,
                        'correct_prediction': is_fake  # Should be True for fake images
                    })
                    
                    if success:
                        logger.info(f"✅ Deepfake detection completed for fake image: {Path(img_path).name}")
                        
            except Exception as e:
                logger.error(f"❌ Deepfake test error for {img_path}: {e}")
                results['fake_tests'].append({
                    'image': Path(img_path).name,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    async def test_face_detection_services(self) -> Dict[str, Any]:
        """Test Face Detection Services (MTCNN, YOLO, MediaPipe)"""
        logger.info("👤 Testing Face Detection Services...")
        results = {
            'mtcnn': [],
            'yolo': [],
            'mediapipe': []
        }
        
        test_images = self.get_test_images()
        
        # Test images with faces
        test_imgs = test_images['group'] + test_images['test'][:2]
        
        for service_name in ['mtcnn', 'yolo', 'mediapipe']:
            port = self.services[service_name]
            
            for img_path in test_imgs:
                try:
                    start_time = time.time()
                    image_data = self.encode_image_to_base64(img_path)
                    
                    payload = {'image': image_data}
                    
                    url = f"{self.base_url}:{port}/detect_faces"
                    async with self.session.post(url, json=payload) as response:
                        response_time = time.time() - start_time
                        data = await response.json()
                        
                        success = response.status == 200
                        faces_count = len(data.get('faces', [])) if success else 0
                        
                        results[service_name].append({
                            'image': Path(img_path).name,
                            'success': success,
                            'response_time': response_time,
                            'faces_detected': faces_count,
                            'faces_data': data.get('faces', [])
                        })
                        
                        if success:
                            logger.info(f"✅ {service_name.upper()} detected {faces_count} faces in {Path(img_path).name}")
                        else:
                            logger.error(f"❌ {service_name.upper()} detection failed for {Path(img_path).name}")
                            
                except Exception as e:
                    logger.error(f"❌ {service_name.upper()} detection error for {img_path}: {e}")
                    results[service_name].append({
                        'image': Path(img_path).name,
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    async def test_gender_age_service(self) -> Dict[str, Any]:
        """Test Gender & Age Detection Service"""
        logger.info("👫 Testing Gender & Age Detection Service...")
        results = []
        
        test_images = self.get_test_images()
        port = self.services['gender_age']
        
        test_imgs = test_images['test'][:3] + test_images['real'][:2]
        
        for img_path in test_imgs:
            try:
                start_time = time.time()
                image_data = self.encode_image_to_base64(img_path)
                
                payload = {'image': image_data}
                
                url = f"{self.base_url}:{port}/analyze"
                async with self.session.post(url, json=payload) as response:
                    response_time = time.time() - start_time
                    data = await response.json()
                    
                    success = response.status == 200
                    
                    results.append({
                        'image': Path(img_path).name,
                        'success': success,
                        'response_time': response_time,
                        'predictions': data.get('predictions', []),
                        'faces_analyzed': len(data.get('predictions', []))
                    })
                    
                    if success:
                        faces_count = len(data.get('predictions', []))
                        logger.info(f"✅ Gender & Age analysis completed for {Path(img_path).name} ({faces_count} faces)")
                    else:
                        logger.error(f"❌ Gender & Age analysis failed for {Path(img_path).name}")
                        
            except Exception as e:
                logger.error(f"❌ Gender & Age analysis error for {img_path}: {e}")
                results.append({
                    'image': Path(img_path).name,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    async def test_api_gateway(self) -> Dict[str, Any]:
        """Test API Gateway Integration"""
        logger.info("🌐 Testing API Gateway...")
        results = {'health_check': None, 'service_routes': []}
        
        port = self.services['api_gateway']
        
        # Test gateway health
        try:
            url = f"{self.base_url}:{port}/health"
            async with self.session.get(url) as response:
                data = await response.json()
                results['health_check'] = {
                    'success': response.status == 200,
                    'status_code': response.status,
                    'data': data
                }
                
                if response.status == 200:
                    logger.info("✅ API Gateway is healthy")
                else:
                    logger.error(f"❌ API Gateway health check failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"❌ API Gateway health check error: {e}")
            results['health_check'] = {'success': False, 'error': str(e)}
        
        # Test service routes through gateway
        test_routes = [
            '/api/face-recognition/health',
            '/api/antispoof/health',
            '/api/deepfake/health',
            '/api/gender-age/health'
        ]
        
        for route in test_routes:
            try:
                url = f"{self.base_url}:{port}{route}"
                async with self.session.get(url) as response:
                    data = await response.text()
                    
                    results['service_routes'].append({
                        'route': route,
                        'success': response.status == 200,
                        'status_code': response.status,
                        'response_time': response.headers.get('X-Response-Time', 'N/A')
                    })
                    
                    if response.status == 200:
                        logger.info(f"✅ Gateway route working: {route}")
                    else:
                        logger.error(f"❌ Gateway route failed: {route} - {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Gateway route error for {route}: {e}")
                results['service_routes'].append({
                    'route': route,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    async def test_end_to_end_pipeline(self) -> Dict[str, Any]:
        """Test complete end-to-end pipeline"""
        logger.info("🔄 Testing End-to-End Pipeline...")
        results = []
        
        test_images = self.get_test_images()
        
        # Use one test image for complete pipeline
        if test_images['test']:
            img_path = test_images['test'][0]
            image_data = self.encode_image_to_base64(img_path)
            
            pipeline_result = {
                'image': Path(img_path).name,
                'steps': {},
                'overall_success': True,
                'total_time': 0
            }
            
            start_total = time.time()
            
            # Step 1: Face Detection (MediaPipe)
            try:
                start_time = time.time()
                payload = {'image': image_data}
                url = f"{self.base_url}:{self.services['mediapipe']}/detect_faces"
                
                async with self.session.post(url, json=payload) as response:
                    step_time = time.time() - start_time
                    data = await response.json()
                    
                    pipeline_result['steps']['face_detection'] = {
                        'success': response.status == 200,
                        'response_time': step_time,
                        'faces_count': len(data.get('faces', [])),
                        'service': 'mediapipe'
                    }
                    
                    if response.status != 200:
                        pipeline_result['overall_success'] = False
                        
            except Exception as e:
                pipeline_result['steps']['face_detection'] = {
                    'success': False,
                    'error': str(e)
                }
                pipeline_result['overall_success'] = False
            
            # Step 2: Anti-Spoofing
            try:
                start_time = time.time()
                payload = {'image': image_data}
                url = f"{self.base_url}:{self.services['antispoof']}/detect_spoof"
                
                async with self.session.post(url, json=payload) as response:
                    step_time = time.time() - start_time
                    data = await response.json()
                    
                    pipeline_result['steps']['antispoof'] = {
                        'success': response.status == 200,
                        'response_time': step_time,
                        'is_real': data.get('is_real'),
                        'confidence': data.get('confidence')
                    }
                    
                    if response.status != 200:
                        pipeline_result['overall_success'] = False
                        
            except Exception as e:
                pipeline_result['steps']['antispoof'] = {
                    'success': False,
                    'error': str(e)
                }
                pipeline_result['overall_success'] = False
            
            # Step 3: Face Recognition
            try:
                start_time = time.time()
                payload = {'image': image_data}
                url = f"{self.base_url}:{self.services['face_recognition']}/recognize"
                
                async with self.session.post(url, json=payload) as response:
                    step_time = time.time() - start_time
                    data = await response.json()
                    
                    pipeline_result['steps']['face_recognition'] = {
                        'success': response.status == 200,
                        'response_time': step_time,
                        'matches': data.get('matches', [])
                    }
                    
                    if response.status != 200:
                        pipeline_result['overall_success'] = False
                        
            except Exception as e:
                pipeline_result['steps']['face_recognition'] = {
                    'success': False,
                    'error': str(e)
                }
                pipeline_result['overall_success'] = False
            
            # Step 4: Gender & Age Detection
            try:
                start_time = time.time()
                payload = {'image': image_data}
                url = f"{self.base_url}:{self.services['gender_age']}/analyze"
                
                async with self.session.post(url, json=payload) as response:
                    step_time = time.time() - start_time
                    data = await response.json()
                    
                    pipeline_result['steps']['gender_age'] = {
                        'success': response.status == 200,
                        'response_time': step_time,
                        'predictions': data.get('predictions', [])
                    }
                    
                    if response.status != 200:
                        pipeline_result['overall_success'] = False
                        
            except Exception as e:
                pipeline_result['steps']['gender_age'] = {
                    'success': False,
                    'error': str(e)
                }
                pipeline_result['overall_success'] = False
            
            pipeline_result['total_time'] = time.time() - start_total
            results.append(pipeline_result)
            
            if pipeline_result['overall_success']:
                logger.info(f"✅ End-to-end pipeline completed successfully in {pipeline_result['total_time']:.2f}s")
            else:
                logger.error(f"❌ End-to-end pipeline failed")
        
        return results
    
    def calculate_test_statistics(self):
        """Calculate overall test statistics"""
        total_tests = 0
        passed_tests = 0
        
        # Count service health tests
        for service, status in self.test_results['service_status'].items():
            total_tests += 1
            if status.get('status') == 'healthy':
                passed_tests += 1
        
        # Count detailed test results
        for test_category, results in self.test_results['detailed_results']:
            if isinstance(results, dict):
                for sub_category, sub_results in results.items():
                    if isinstance(sub_results, list):
                        for result in sub_results:
                            total_tests += 1
                            if result.get('success', False):
                                passed_tests += 1
            elif isinstance(results, list):
                for result in results:
                    total_tests += 1
                    if result.get('success', False):
                        passed_tests += 1
        
        self.test_results['total_tests'] = total_tests
        self.test_results['passed_tests'] = passed_tests
        self.test_results['failed_tests'] = total_tests - passed_tests
        
        # Calculate performance metrics
        response_times = []
        for _, results in self.test_results['detailed_results']:
            if isinstance(results, dict):
                for sub_results in results.values():
                    if isinstance(sub_results, list):
                        for result in sub_results:
                            if 'response_time' in result:
                                response_times.append(result['response_time'])
            elif isinstance(results, list):
                for result in results:
                    if 'response_time' in result:
                        response_times.append(result['response_time'])
        
        if response_times:
            self.test_results['performance_metrics'] = {
                'avg_response_time': sum(response_times) / len(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'total_requests': len(response_times)
            }
    
    def generate_report(self):
        """Generate comprehensive test report"""
        report = []
        report.append("=" * 80)
        report.append("FACESOCIAL AI SERVICES - COMPLETE SYSTEM TEST REPORT")
        report.append("=" * 80)
        report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Tests: {self.test_results['total_tests']}")
        report.append(f"Passed: {self.test_results['passed_tests']}")
        report.append(f"Failed: {self.test_results['failed_tests']}")
        
        if self.test_results['total_tests'] > 0:
            success_rate = (self.test_results['passed_tests'] / self.test_results['total_tests']) * 100
            report.append(f"Success Rate: {success_rate:.1f}%")
        
        report.append("")
        
        # Service Status Summary
        report.append("SERVICE STATUS SUMMARY:")
        report.append("-" * 40)
        for service, status in self.test_results['service_status'].items():
            status_icon = "✅" if status.get('status') == 'healthy' else "❌"
            report.append(f"{status_icon} {service.upper()}: {status.get('status', 'unknown')}")
        
        report.append("")
        
        # Performance Metrics
        if self.test_results.get('performance_metrics'):
            metrics = self.test_results['performance_metrics']
            report.append("PERFORMANCE METRICS:")
            report.append("-" * 40)
            report.append(f"Average Response Time: {metrics['avg_response_time']:.3f}s")
            report.append(f"Min Response Time: {metrics['min_response_time']:.3f}s")
            report.append(f"Max Response Time: {metrics['max_response_time']:.3f}s")
            report.append(f"Total Requests: {metrics['total_requests']}")
            report.append("")
        
        # Detailed Results
        report.append("DETAILED TEST RESULTS:")
        report.append("-" * 40)
        for test_name, results in self.test_results['detailed_results']:
            report.append(f"\n{test_name.upper()}:")
            if isinstance(results, dict):
                for category, category_results in results.items():
                    if isinstance(category_results, list):
                        passed = sum(1 for r in category_results if r.get('success', False))
                        total = len(category_results)
                        report.append(f"  {category}: {passed}/{total} passed")
                    else:
                        status = "✅" if category_results.get('success', False) else "❌"
                        report.append(f"  {category}: {status}")
            elif isinstance(results, list):
                passed = sum(1 for r in results if r.get('success', False))
                total = len(results)
                report.append(f"  Tests: {passed}/{total} passed")
        
        report.append("")
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    async def run_complete_test_suite(self):
        """Run the complete test suite"""
        logger.info("🚀 Starting FaceSocial Complete System Test...")
        
        await self.setup_session()
        
        try:
            # 1. Test Service Health
            logger.info("📊 Testing Service Health...")
            for service_name, port in self.services.items():
                await self.test_service_health(service_name, port)
            
            # 2. Test Individual Services
            test_results = []
            
            # Face Recognition Service
            face_recognition_results = await self.test_face_recognition_service()
            test_results.append(("Face Recognition Service", face_recognition_results))
            
            # Anti-Spoofing Service
            antispoof_results = await self.test_antispoof_service()
            test_results.append(("Anti-Spoofing Service", antispoof_results))
            
            # Deepfake Detection Service
            deepfake_results = await self.test_deepfake_service()
            test_results.append(("Deepfake Detection Service", deepfake_results))
            
            # Face Detection Services
            face_detection_results = await self.test_face_detection_services()
            test_results.append(("Face Detection Services", face_detection_results))
            
            # Gender & Age Detection Service
            gender_age_results = await self.test_gender_age_service()
            test_results.append(("Gender & Age Detection Service", gender_age_results))
            
            # API Gateway
            api_gateway_results = await self.test_api_gateway()
            test_results.append(("API Gateway", api_gateway_results))
            
            # End-to-End Pipeline
            e2e_results = await self.test_end_to_end_pipeline()
            test_results.append(("End-to-End Pipeline", e2e_results))
            
            self.test_results['detailed_results'] = test_results
            
            # Calculate statistics
            self.calculate_test_statistics()
            
            # Generate and save report
            report = self.generate_report()
            
            # Save report to file
            with open('system_test_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Save detailed JSON results
            with open('system_test_results.json', 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
            
            print(report)
            
            logger.info("✅ System testing completed successfully!")
            logger.info("📄 Reports saved to: system_test_report.txt, system_test_results.json")
            
        except Exception as e:
            logger.error(f"❌ System test failed: {e}")
            traceback.print_exc()
            
        finally:
            await self.cleanup_session()

async def main():
    """Main function to run the complete system test"""
    print("🔍 FaceSocial AI Services - Complete System Test")
    print("=" * 60)
    print("This test will verify all AI services in the FaceSocial project:")
    print("• Face Recognition Service")
    print("• Anti-Spoofing Service")
    print("• Deepfake Detection Service")
    print("• Face Detection Services (MTCNN, YOLO, MediaPipe)")
    print("• Gender & Age Detection Service")
    print("• API Gateway Integration")
    print("• End-to-End Pipeline Testing")
    print("=" * 60)
    
    # Check if test images exist
    test_images_dir = Path("test-image")
    if not test_images_dir.exists():
        print("⚠️  Warning: Test images directory not found!")
        print("Please ensure test-image directory exists with sample images.")
        print("Expected image types: real_*.jpg, fake_*.jpg, spoof_*.jpg, group_*.jpg, test_*.jpg")
        print("")
    
    print("🚦 Make sure all services are running:")
    print("• docker-compose up -d (for all services)")
    print("• Or start individual services on ports: 5001-5007, 8080")
    print("")
    
    input("Press Enter to start testing... ")
    
    tester = FaceSocialSystemTester()
    await tester.run_complete_test_suite()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        traceback.print_exc()
