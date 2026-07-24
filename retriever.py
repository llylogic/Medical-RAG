import os
import pickle
import jieba
import numpy as np
from typing import List, Tuple
from core.embedding_ops import Embedder
from core.vector_store import VectorStore

class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder, bm25_path: str = "rag_interview_project/cache/bm25_index.pkl"):
        self.vector_store = vector_store
        self.embedder = embedder
        
        # 仅加载，不构建，提速 100 倍
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.bm25_docs = data["docs"] 
        else:
            print("警告：未找到本地 BM25 索引，只允许 Dense 检索")
            self.bm25 = None
            self.bm25_docs = []

    def _dense_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        query_emb = self.embedder.encode([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_emb], n_results=top_k
        )
        if not results['documents'][0]: return []
        return list(zip(results['documents'][0], results['distances'][0]))

    def _sparse_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not self.bm25: return []
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        top_n_idx = np.argsort(scores)[::-1][:top_k]
        # 必须从缓存的文档列表取，绝对不能依赖 ChromaDB 顺序
        return [(self.bm25_docs[i], float(scores[i])) for i in top_n_idx]

    def search(self, query: str, top_k: int = 10, strategy: str = "hybrid") -> List[str]:
        # hybrid 逻辑利用RRF 算法
        if strategy == "dense":
            return [doc for doc, _ in self._dense_search(query, top_k)]
        elif strategy == "bm25":
            return [doc for doc, _ in self._sparse_search(query, top_k)]
        elif strategy == "hybrid":
            dense_res = self._dense_search(query, top_k * 2)
            sparse_res = self._sparse_search(query, top_k * 2)
            rrf_scores = {}
            k_rrf = 60
            for rank, (doc, _) in enumerate(dense_res):
                rrf_scores[doc] = rrf_scores.get(doc, 0) + 1.0 / (k_rrf + rank + 1)
            for rank, (doc, _) in enumerate(sparse_res):
                rrf_scores[doc] = rrf_scores.get(doc, 0) + 1.0 / (k_rrf + rank + 1)
            sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in sorted_docs[:top_k]]