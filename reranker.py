# 硅基流动 Reranker
import requests
from typing import List

class Reranker:
    def __init__(self, api_key: str, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.api_key = api_key
        self.url = "https://api.siliconflow.cn/v1/rerank"

    def rerank(self, query: str, docs: List[str],  top_n: int = 3, threshold: float = 0.0) -> List[str]:
        
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": docs,
            "top_n": len(docs), # 先给所有文档打分，别急着截断
            "return_documents": True
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(self.url, json=payload, headers=headers)
        if response.status_code == 200:
            results = response.json().get("results", [])
            # ================= 核心亮点：置信度截断 =================
            filtered_docs = []
            for res in results:
                # 提取模型给出的相关度绝对打分
                score = res.get("relevance_score", 0.0)
                # 只有超过阈值（及格）的文献，才有资格留下来
                if score >= threshold:
                    filtered_docs.append(res["document"]["text"])
                    
            # 最后再取 Top-N
            return filtered_docs[:top_n]
        else:
            print(f"Rerank API 失败: {response.text}")
            return docs[:top_n]