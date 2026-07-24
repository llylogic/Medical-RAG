# 硅基流动 Embedder
from typing import List
from openai import OpenAI

class Embedder:
    def __init__(self, api_key: str, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1"
        )

    def encode(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        return [data.embedding for data in response.data]