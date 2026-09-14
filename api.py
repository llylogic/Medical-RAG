"""放在 Medical-RAG 项目根目录。启动：python -m uvicorn api:app --port 8000"""

import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_api_service import RagService

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("medical_rag_api")
bearer = HTTPBearer(auto_error=False)


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=8000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def check_total_length(self):
        if len(self.message) + sum(len(m.content) for m in self.history) > 32000:
            raise ValueError("问题与历史记录总长度不能超过 32000 个字符")
        return self


class Reference(BaseModel):
    title: str
    content: str


class ChatResponse(BaseModel):
    answer: str
    references: list[Reference]


def verify_key(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
):
    valid = credentials is not None and secrets.compare_digest(
        credentials.credentials.encode("utf-8"),
        request.app.state.api_key.encode("utf-8"),
    )
    if not valid:
        raise HTTPException(
            status_code=401,
            detail="API Key 缺失或无效",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_app(service=None, api_key=None):
    @asynccontextmanager
    async def lifespan(application):
        load_dotenv(ROOT / ".env.api", override=False)
        key = api_key if api_key is not None else os.getenv("APP_API_KEY", "")
        if len(key.strip()) < 32 or "请替换" in key:
            raise RuntimeError("请在 .env.api 中设置至少 32 字符的 APP_API_KEY")
        application.state.api_key = key
        application.state.rag = service if service is not None else RagService.from_env(ROOT)
        # 首版每个进程同时处理一条 RAG 请求，避免重叠执行耗尽上游资源。
        application.state.slot = threading.BoundedSemaphore(1)
        yield
        if service is None:
            application.state.rag.close()

    application = FastAPI(
        title="Medical-RAG 对话 API",
        description="用自己的 API Key 调用整个 RAG。此版本为非流式自定义 /chat 协议。",
        lifespan=lifespan,
    )

    @application.get("/health")
    def health():
        return {"status": "ok"}

    @application.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_key)])
    def chat(payload: ChatRequest, request: Request):
        slot = request.app.state.slot
        if not slot.acquire(blocking=False):
            raise HTTPException(429, "服务正在处理其他请求，请稍后重试", headers={"Retry-After": "5"})
        try:
            history = [message.model_dump() for message in payload.history]
            return request.app.state.rag.ask(payload.message, history)
        except Exception as exc:
            error_id = secrets.token_hex(6)
            # 不把供应商原始错误、密钥、问题和资料内容返回给客户或写入这里的日志。
            logger.error("RAG request failed: id=%s type=%s", error_id, type(exc).__name__)
            raise HTTPException(502, f"RAG 处理失败，请联系服务提供方，错误编号：{error_id}") from None
        finally:
            slot.release()

    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
