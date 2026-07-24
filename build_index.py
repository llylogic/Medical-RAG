# 【离线】知识库全量构建入口
import os
from tqdm import tqdm
from core.document_ops import DocumentLoader, TextCleaner, TextChunker
from core.embedding_ops import Embedder
from core.vector_store import VectorStore
import pickle
from rank_bm25 import BM25Okapi
import jieba
import time

SILICONFLOW_API_KEY = "你的真实硅基流动key"

def build_index():
    print("=== 开始构建医疗离线知识库 ===")
    loader = DocumentLoader()
    cleaner = TextCleaner()
    chunker = TextChunker(chunk_size=1000)
    
    raw_texts = loader.load_directory("rag_interview_project/data")
    all_chunks = []
    for text in raw_texts:
        cleaned = cleaner.clean(text)
        chunks = chunker.split(cleaned)
        all_chunks.extend(chunks)
        
    print(f"成功切分为 {len(all_chunks)} 个带疾病上下文的知识块。")

    embedder = Embedder(api_key=SILICONFLOW_API_KEY)
    vector_store = VectorStore(persist_dir="rag_interview_project/chroma_db")
    
    batch_size = 50
    print("正在请求硅基流动 API 进行向量化并入库 ChromaDB...")
    
    for i in tqdm(range(0, len(all_chunks), batch_size)):
        batch_chunks = all_chunks[i : i + batch_size]
        
        # 【面试亮点：工程鲁棒性 - 加入指数退避重试机制应对限流】
        max_retries = 5
        for attempt in range(max_retries):
            try:
                batch_embeddings = embedder.encode(batch_chunks)
                # 使用 Upsert 机制（你之前已经改过了），所以重试插入非常安全
                vector_store.store(batch_chunks, batch_embeddings)
                break  # 成功则跳出重试循环
                
            except Exception as e:
                # 捕捉到 429 限流报错
                if "429" in str(e) or "TPM limit" in str(e):
                    wait_time = 15 * (attempt + 1)  # 渐进式等待: 15s, 30s, 45s...
                    print(f"\n⚠️ 触发API限流 (TPM超载)，暂停 {wait_time} 秒后重试 (第 {attempt+1}/{max_retries} 次)...")
                    time.sleep(wait_time)
                else:
                    # 如果是其他严重报错，就抛出
                    raise e
                    
        # 正常请求成功后，也稍微停顿 1 秒，细水长流防止触发限流
        time.sleep(1)
        
    print("正在构建并持久化 BM25 词频索引...")
    # 【面试亮点：BM25与文档严格对齐打包存储，避免 ChromaDB 顺序错乱】
    tokenized_corpus = [list(jieba.cut(doc)) for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    os.makedirs("rag_interview_project/cache", exist_ok=True)
    with open("rag_interview_project/cache/bm25_index.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "docs": all_chunks}, f)
        
    print("=== 离线知识库构建完成！ ===")

if __name__ == "__main__":
    build_index()