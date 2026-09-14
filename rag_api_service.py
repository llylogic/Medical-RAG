"""复用项目核心模块，独立于 Gradio；只在服务启动时连接已有索引。"""

import os
import re
from pathlib import Path


PREFIX = re.compile(r"^\[关联疾病:\s*(.*?)\s*\|\s*序号:\s*(\d+)\]\s*")


def merge_references(documents):
    groups = {}
    references = []
    for document in documents:
        text = document.strip()
        match = PREFIX.match(text)
        if match:
            name = match.group(1).strip()
            index = int(match.group(2))
            content = text[match.end():].strip()
            if name not in groups:
                groups[name] = {}
                references.append((name, groups[name]))
            groups[name][index] = content
        elif text:
            references.append(("知识库片段", {0: text}))
    return [
        {"title": name, "content": "\n".join(chunks[i] for i in sorted(chunks))}
        for name, chunks in references
    ]


class RagService:
    def __init__(self, retriever, reranker, llm, embedder=None):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.embedder = embedder

    @classmethod
    def from_env(cls, root: Path):
        required = ("SILICONFLOW_API_KEY", "DEEPSEEK_API_KEY", "RAG_CHROMA_DIR", "RAG_BM25_PATH")
        missing = [name for name in required if not os.getenv(name) or "请替换" in os.getenv(name, "")]
        if missing:
            raise RuntimeError("请配置 .env.api：" + ", ".join(missing))

        def resolve_path(value):
            path = Path(value).expanduser()
            return path if path.is_absolute() else root / path

        chroma_dir = resolve_path(os.environ["RAG_CHROMA_DIR"])
        bm25_path = resolve_path(os.environ["RAG_BM25_PATH"])
        if not (chroma_dir / "chroma.sqlite3").is_file():
            raise RuntimeError("RAG_CHROMA_DIR 应指向已有索引中包含 chroma.sqlite3 的目录")
        if not bm25_path.is_file():
            raise RuntimeError("RAG_BM25_PATH 应指向已有的 bm25_index.pkl 文件")

        # 与项目 web_app.py 的导入方式一致。按服务器实际布局调整这五行即可。
        from core.embedding_ops import Embedder
        from core.vector_store import VectorStore
        from core.retriever import Retriever
        from core.reranker import Reranker
        from core.llm_generator import LLMGenerator

        silicon_key = os.environ["SILICONFLOW_API_KEY"]
        embedder = Embedder(
            api_key=silicon_key,
            model_name=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3"),
        )
        vector_store = VectorStore(persist_dir=str(chroma_dir))
        if vector_store.collection.count() == 0:
            raise RuntimeError("medical_knowledge_base 集合为空，请检查已有索引的路径")
        retriever = Retriever(vector_store, embedder, bm25_path=str(bm25_path))
        reranker = Reranker(
            api_key=silicon_key,
            model_name=os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        )
        llm = LLMGenerator(api_key=os.environ["DEEPSEEK_API_KEY"])
        return cls(retriever, reranker, llm, embedder)

    def ask(self, message, history):
        # 每个请求自己携带历史，服务端没有所有客户共用的全局聊天记录。
        history = [dict(item) for item in history]
        search_query = self.llm.rewrite_query(message, history)
        documents = self.retriever.search(query=search_query, top_k=20, strategy="hybrid")
        if documents:
            documents = self.reranker.rerank(
                query=search_query, docs=documents, top_n=5, threshold=float(os.getenv("RAG_RERANK_THRESHOLD", "0.3"))
            )
        references = merge_references(documents)
        context_docs = [f"《{item['title']}》检索资料：\n{item['content']}" for item in references]

        # 原生成器每次 yield 的是累计答案；取最后一次，不要把每次输出拼接起来。
        answer = ""
        for partial in self.llm.generate_multi_turn_stream(message, context_docs, history):
            answer = partial
        if not answer.strip():
            raise RuntimeError("生成器没有返回答案")
        return {"answer": answer, "references": references}

    def close(self):
        for component in (self.embedder, self.llm):
            client = getattr(component, "client", None)
            if client is not None:
                client.close()
