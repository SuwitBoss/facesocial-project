from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class MilvusManager:
    def __init__(self, host="milvus", port="19530"):
        self.host = host
        self.port = port
        self.collection_name = "face_embeddings"
        self.collection = None
        self._connect()
        self._create_collection()
    
    def _connect(self):
        """Connect to Milvus"""
        try:
            connections.connect(alias="default", host=self.host, port=self.port)
            logger.info("Connected to Milvus successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _create_collection(self):
        """Create face embeddings collection if not exists"""
        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
            else:
                # Define schema
                fields = [
                    FieldSchema(name="face_id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
                    FieldSchema(name="model_type", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="created_at", dtype=DataType.INT64)
                ]
                
                schema = CollectionSchema(fields, description="Face embeddings for recognition")
                self.collection = Collection(self.collection_name, schema)
                
                # Create index
                index_params = {
                    "metric_type": "IP",  # Inner Product for normalized vectors
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                self.collection.create_index("embedding", index_params)
                
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def insert_face(self, user_id: str, face_id: str, embedding: np.ndarray, model_type: str = "ensemble") -> bool:
        """Insert face embedding into Milvus"""
        try:
            import time
            
            data = [
                [face_id],
                [user_id],
                [embedding.tolist()],
                [model_type],
                [int(time.time())]
            ]
            
            self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Inserted face {face_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert face: {e}")
            return False
    
    def search_faces(self, embedding: np.ndarray, top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        """Search for similar faces"""
        try:
            self.collection.load()
            
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }
            
            results = self.collection.search(
                data=[embedding.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["user_id", "face_id", "model_type"]
            )
            
            matches = []
            for hits in results:
                for hit in hits:
                    if hit.score >= threshold:
                        matches.append({
                            "user_id": hit.entity.get("user_id"),
                            "face_id": hit.entity.get("face_id"),
                            "similarity": float(hit.score),
                            "model_type": hit.entity.get("model_type")
                        })
            
            return matches
            
        except Exception as e:
            logger.error(f"Failed to search faces: {e}")
            return []
