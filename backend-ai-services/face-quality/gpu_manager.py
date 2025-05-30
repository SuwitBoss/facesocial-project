"""
GPU Memory Manager for Face Quality Service
"""

import os
import torch
import GPUtil
import psutil
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """Manage GPU memory for quality assessment service"""
    
    def __init__(self, memory_limit_mb: int = 400):
        self.memory_limit_mb = memory_limit_mb
        self.allocated_memory = 0
        
    def check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability and memory status"""
        gpu_info = {
            "gpu_available": False,
            "gpu_count": 0,
            "devices": []
        }
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info["gpu_available"] = True
                gpu_info["gpu_count"] = len(gpus)
                
                for i, gpu in enumerate(gpus):
                    device_info = {
                        "id": i,
                        "name": gpu.name,
                        "memory_total_mb": gpu.memoryTotal,
                        "memory_used_mb": gpu.memoryUsed,
                        "memory_free_mb": gpu.memoryFree,
                        "memory_util_percent": gpu.memoryUtil * 100,
                        "gpu_util_percent": gpu.load * 100,
                        "temperature": gpu.temperature
                    }
                    gpu_info["devices"].append(device_info)
                    
        except Exception as e:
            logger.error(f"Error checking GPU: {e}")
            
        return gpu_info
        
    def allocate_memory(self, size_mb: int) -> bool:
        """Try to allocate GPU memory"""
        gpu_info = self.check_gpu_availability()
        
        if not gpu_info["gpu_available"]:
            return False
            
        gpu = gpu_info["devices"][0]
        free_memory = gpu["memory_free_mb"]
        
        if size_mb <= free_memory and (self.allocated_memory + size_mb) <= self.memory_limit_mb:
            self.allocated_memory += size_mb
            logger.info(f"Allocated {size_mb}MB GPU memory. Total: {self.allocated_memory}MB")
            return True
        else:
            logger.warning(f"Cannot allocate {size_mb}MB. Free: {free_memory}MB, Limit: {self.memory_limit_mb}MB")
            return False
            
    def release_memory(self, size_mb: int):
        """Release GPU memory"""
        self.allocated_memory = max(0, self.allocated_memory - size_mb)
        logger.info(f"Released {size_mb}MB GPU memory. Total: {self.allocated_memory}MB")
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Global GPU manager instance
gpu_manager = GPUMemoryManager(memory_limit_mb=int(os.getenv('CUDA_MEM_LIMIT', '400').rstrip('m')))