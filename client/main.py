"""在客户电脑运行：python main.py。无需安装 FastAPI 或 RAG 依赖。"""

import getpass

from rag_client import ClientError, RagClient


def recent_history(history, message):
    """遵守当前服务端限制，超限时从最早的一整轮问答开始移除。"""
    selected = list(history)
    while selected and (
        len(selected) > 20
        or len(message) + sum(len(item["content"]) for item in selected) > 32000
        or any(len(item["content"]) > 8000 for item in selected)
    ):
        selected = selected[2:]
    return selected


def show_references(references):
    if not references:
        print("还没有可显示的参考资料。")
    for index, item in enumerate(references, 1):
        print("\n[{}] {}\n{}".format(index, item["title"], item["content"]))


def main():
    print("RAG 客户端：通过 HTTP 接口向服务器提问。")
    print("服务地址不包含 /docs 或 /chat；直接回车使用默认地址。")
    base_url = input("服务地址 [http://127.0.0.1:8000]：").strip() or "http://127.0.0.1:8000"
    api_key = getpass.getpass("服务方发给你的 Key（输入不显示，粘贴后按回车）：")
    try:
        client = RagClient(base_url, api_key)
    except ClientError as exc:
        print(exc)
        return 1

    print("\n输入问题即可发送；/refs 查看资料，/new 新对话，/health 检查连接，/quit 退出。")
    print("健康检查不验证 Key；第一次问答成功才表明 Key 和问答调用都通过。")
    history, references = [], []
    while True:
        message = input("\n你：").strip()
        if not message:
            continue
        if message == "/quit":
            return 0
        if message == "/new":
            history, references = [], []
            print("已开始新对话。")
            continue
        if message == "/refs":
            show_references(references)
            continue
        if message == "/health":
            try:
                client.health()
                print("连接成功：接口返回 status=ok。此检查没有调用模型。")
            except ClientError as exc:
                print(exc)
            continue
        if message.startswith("/"):
            print("可用命令：/refs、/new、/health、/quit。")
            continue
        if len(message) > 8000:
            print("问题不能超过 8000 个字符，请缩短后重试。")
            continue
        selected = recent_history(history, message)
        if len(selected) < len(history):
            print("为符合接口长度限制，本次只携带能够保留的最近完整问答轮次。")
        print("正在等待服务器回答……")
        try:
            result = client.chat(message, selected)
        except ClientError as exc:
            print(exc)
            continue
        print("\n回答：\n" + result["answer"])
        references = result["references"]
        print("\n收到 {} 条参考资料，输入 /refs 查看原文。".format(len(references)))
        history = selected + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["answer"]},
        ]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\n客户端已退出。服务器服务和 SSH 通道不会因此关闭。")
