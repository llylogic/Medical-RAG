"""客户方的 HTTP 调用代码。只使用 Python 标准库，不导入任何服务端模块。"""

import json
import socket
import urllib.error
import urllib.parse
import urllib.request


class ClientError(Exception):
    """可以直接展示给使用者的错误。"""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # 不把带 Key 的请求自动转发到重定向后的地址。
        return None


class RagClient:
    def __init__(self, base_url, api_key, timeout=180):
        self.base_url = base_url.strip().rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ClientError("接口地址应类似 http://127.0.0.1:8000")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ClientError("地址中不要包含账号、密码、查询参数或 # 后面的内容。")
        if parsed.path.rstrip("/").endswith(("/docs", "/chat", "/health", "/openapi.json")):
            raise ClientError("请填写服务根地址，例如 http://127.0.0.1:8000，不要加 /docs 或 /chat。")
        self.api_key = api_key.strip()
        if not self.api_key or not self.api_key.isascii() or any(c.isspace() for c in self.api_key):
            raise ClientError("请粘贴服务方提供的 APP_API_KEY 值，不要加引号或 Bearer。")
        self.timeout = timeout
        handlers = [NoRedirect()]
        if parsed.hostname in ("127.0.0.1", "localhost", "::1"):
            # SSH 转发的本机地址直接连接，避免被电脑上的系统代理带走。
            handlers.append(urllib.request.ProxyHandler({}))
        self.opener = urllib.request.build_opener(*handlers)

    def _request(self, path, payload=None):
        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers.update({
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
            })
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            messages = {
                401: "Key 缺失或无效。请使用服务方的 APP_API_KEY，不是硅基流动或 DeepSeek 的 Key。",
                404: "找不到接口，请检查服务地址和端口是否正确。",
                422: "请求格式或长度不符合接口要求。",
                429: "服务正在处理其他请求，请稍后再试。",
                502: "请求已到达服务方，但 RAG 执行失败。请查看服务端终端中的错误编号。",
            }
            explanation = messages.get(exc.code, "服务返回错误，请联系服务提供方。")
            if 300 <= exc.code < 400:
                explanation = "地址发生重定向，请向服务方确认最终接口地址后再连接。"
            exc.close()
            raise ClientError("HTTP {}：{}".format(exc.code, explanation)) from None
        except (TimeoutError, socket.timeout):
            raise ClientError("等待响应超时；服务端可能仍在生成答案。请先查看服务端状态，再决定是否重试。") from None
        except urllib.error.URLError:
            raise ClientError("无法连接接口。请检查服务是否运行、地址是否正确，以及本地端口转发是否保持连接。") from None
        except (ValueError, UnicodeError):
            raise ClientError("响应不是有效的 JSON，请确认这个地址运行的是 RAG API。") from None

    def health(self):
        result = self._request("/health")
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise ClientError("健康检查返回了意外内容，请确认接口地址。")
        return result

    def chat(self, message, history=None):
        # 客户端只提交文本；检索索引和模型调用都在服务端进行。
        result = self._request("/chat", {"message": message, "history": history or []})
        if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
            raise ClientError("响应中缺少 answer 文本，请检查接口版本。")
        references = result.get("references")
        if not isinstance(references, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("title"), str)
            or not isinstance(item.get("content"), str)
            for item in references
        ):
            raise ClientError("响应中的 references 格式不符合约定。")
        return result
