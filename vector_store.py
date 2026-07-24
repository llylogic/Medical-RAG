# ChromaDB 存储
import sys
from unittest.mock import MagicMock
# [魔法代码] 欺骗 Python 解释器，塞入一个假的 onnxruntime 空壳
sys.modules['onnxruntime'] = MagicMock()

import chromadb
import hashlib
from typing import List

class VectorStore:
    def __init__(self, persist_dir: str = "rag_interview_project/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("medical_knowledge_base")

    def store(self, chunks: List[str], embeddings: List[List[float]]) -> None:
        # 【面试亮点：利用 MD5 实现幂等性写入，防止同一份文档重复入库】
        ids = []
        for chunk in chunks:
            chunk_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()
            ids.append(f"doc_{chunk_hash}")
            
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=chunks)
    
    def get_all_documents(self) -> List[str]:
        results = self.collection.get(include=['documents'])
        return results['documents'] if results['documents'] else []