import re
import gradio as gr
from core.embedding_ops import Embedder
from core.vector_store import VectorStore
from core.retriever import Retriever
from core.reranker import Reranker
from core.llm_generator import LLMGenerator

SILICONFLOW_API_KEY = "你的真实硅基流动key"
DEEPSEEK_API_KEY = "你的真实DeepSeekkey"

print("正在初始化系统组件，请稍候...")
embedder = Embedder(api_key=SILICONFLOW_API_KEY)
vector_store = VectorStore(persist_dir="rag_interview_project/chroma_db")
retriever = Retriever(vector_store, embedder, bm25_path="rag_interview_project/cache/bm25_index.pkl")
reranker = Reranker(api_key=SILICONFLOW_API_KEY)
llm = LLMGenerator(api_key=DEEPSEEK_API_KEY)
print("初始化完成！准备启动Web界面...")

# ================= 自定义高级 CSS 样式 =================
custom_css = """
body { background-color: #f3f4f6; font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif; }
.header-banner { 
    background: linear-gradient(135deg, #0072ff 0%, #00c6ff 100%); 
    color: white; padding: 25px; border-radius: 12px; 
    text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
}
.header-banner h1 { margin: 0; font-size: 28px; font-weight: 600; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }
.header-banner p { margin: 10px 0 0 0; font-size: 15px; opacity: 0.9; }
.panel-box { border: 1px solid #e5e7eb; padding: 20px; border-radius: 12px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
.source-doc { background-color: #f8fafc; border-left: 4px solid #3b82f6; padding: 10px 15px; margin-bottom: 10px; border-radius: 0 8px 8px 0; font-size: 13px; line-height: 1.6; }
"""

# ================= 核心业务逻辑 (分离 User 和 Bot 动作) =================

# ================= 核心业务逻辑 (Gradio 6.0 标准字典流格式) =================

def user_input(user_message, history):
    """用户提交问题，更新 UI 状态"""
    history = history or []
    # 采用标准格式记录用户的提问
    history.append({"role": "user", "content": user_message})
    return "", history

def bot_response(history, top_k_val, top_n_val, threshold_val):
    """机器人的核心 RAG 处理流水线"""
     # ================= 终极护盾：清洗 Gradio 6.0 强制附加的变态格式 =================
    for msg in history:
        # 如果 Gradio 把纯文本变成了 [{'text': 'xxx', 'type': 'text'}] 的多模态格式，我们强行把它提取回纯字符串！
        if isinstance(msg.get("content"), list):
            msg["content"] = " ".join([item.get("text", "") for item in msg["content"] if isinstance(item, dict) and "text" in item])
            
    query = history[-1]["content"]  # 从字典里提取用户最新问题
    past_history = history[:-1]     # 提取不包含当前问题的历史记录
    
    # 1. 查询重写 (在界面上显示“正在思考”的气泡)
    history.append({"role": "assistant", "content": "🔄 正在结合历史对话重写搜索意图..."})
    yield history, "正在分析意图..."
    
    search_query = llm.rewrite_query(query, past_history)
    print(f"\n🔄 LLM 意图重写: [{query}] -> [{search_query}]")

    # 2. 混合检索与重排 (动态展示搜索参数)
    history[-1]["content"] = f"🔍 正在知识库中检索: {search_query} (召回 {top_k_val} 篇, 精排 {top_n_val} 篇)..."
    yield history, "正在检索文献..."
    
    retrieved_docs = retriever.search(query=search_query, top_k=top_k_val, strategy="hybrid")
    reranked_docs = reranker.rerank(query=search_query, docs=retrieved_docs, top_n=top_n_val, threshold=threshold_val)

    # 3. 动态聚合 (带有绝对坐标排序)
    merged_context = {}
    for doc in reranked_docs:
        match = re.search(r'【关联疾病:\s*(.*?)\s*\|\s*序号:\s*(\d+)】', doc)
        if match:
            disease_name = match.group(1).strip()
            chunk_idx = int(match.group(2))
            clean_content = doc.replace(match.group(0), "").strip()
            if disease_name not in merged_context:
                merged_context[disease_name] = []
            merged_context[disease_name].append((chunk_idx, clean_content))
        else:
            merged_context[doc] = [(0, doc)]

    final_context_docs = []
    trace_html = "### 📚 检索到的核心医学依据\n"
    for disease, chunks in merged_context.items():
        if isinstance(chunks, list) and isinstance(chunks[0], tuple):
            chunks.sort(key=lambda x: x[0])
            assembled_content = "\n".join([text for idx, text in chunks])
            final_context_docs.append(f"《{disease}》完整整合资料：\n{assembled_content}")
            # 生成好看的 HTML 溯源证据卡片
            trace_html += f"<div class='source-doc'><b>🩺 关联实体: {disease}</b><br>{assembled_content[:2000]}...</div>"
        else:
            final_context_docs.append(chunks[0][1])
            trace_html += f"<div class='source-doc'>{chunks[0][1][:2000]}...</div>"

    if len(reranked_docs) > len(final_context_docs):        
            print(f"\n✔ 动态聚合成功：将 {len(reranked_docs)} 个乱序切片，按绝对坐标重组为 {len(final_context_docs)} 篇完整上下文！")    
                
    if not final_context_docs:
        trace_html = "⚠️ 未在知识库中检索到高度匹配的文献。"

    # 4. 流式调用大模型
    history[-1]["content"] = "" # 清空之前的提示语，准备逐字输出最终答案
    for partial_answer in llm.generate_multi_turn_stream(query, final_context_docs, past_history):
        history[-1]["content"] = partial_answer
        yield history, trace_html

# ================= 构建 Web UI 布局 =================

with gr.Blocks() as demo:
    
    # 顶部横幅
    gr.HTML("""
    <div class="header-banner">
        <h1>🏥 医疗专家辅助诊疗系统</h1>
    </div>
    """)
    
    with gr.Row():
        # ================= 左侧：控制台面板 (占 1/4) =================
        with gr.Column(scale=1):
            with gr.Group(elem_classes="panel-box"):
                gr.Markdown("### ⚙️ RAG 检索参数调优")
                gr.Markdown("<span style='color:gray; font-size:12px;'>*调节下方参数可实时改变检索漏斗形态*</span>")
                
                slider_top_k = gr.Slider(minimum=5, maximum=50, value=20, step=5, label="粗排召回量 (Top-K)", info="底层 BM25+Dense 双路RRF初筛数量")
                slider_top_n = gr.Slider(minimum=2, maximum=15, value=5, step=1, label="精排截断量 (Top-N)", info="Cross-Encoder 重排后喂给大模型的文献数")
                slider_threshold = gr.Slider(minimum=-2.0, maximum=2.0, value=0.0, step=0.1, label="相似度及格线 (Threshold)", info="得分低于此阈值的文献将被直接拦截丢弃")

            with gr.Group(elem_classes="panel-box", visible=True):
                gr.Markdown("### 🛠️ 提问例子")
                gr.Examples(
                    examples=[
                        "肌纤维组织炎应该怎么预防？",
                        "小儿进行性骨化性肌炎的发病机制是什么？",
                        "功能性聋是什么意思？",
                        "空腹血糖、餐后2小时血糖正常范围分别是多少？"
                    ],
                    inputs=[gr.Textbox(visible=False)], # 占位
                    label=""
                )
                
        # ================= 右侧：交互与溯源面板 (占 3/4) =================
        with gr.Column(scale=3):
            # 聊天框
            chatbot = gr.Chatbot(
                height=500, 
                show_label=False,
                avatar_images=["https://cdn-icons-png.flaticon.com/512/3135/3135715.png", "https://cdn-icons-png.flaticon.com/512/3304/3304260.png"]
            )
            
            # 输入区
            with gr.Row():
                msg_input = gr.Textbox(
                    show_label=False, 
                    placeholder="请输入患者的详细症状或疾病名称，按 Enter 发送...", 
                    scale=8,
                    container=False
                )
                submit_btn = gr.Button("🩺 提交诊疗", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️ 清空", scale=1)

            # 溯源证据折叠面板 (Killer Feature!)
            with gr.Accordion("🔍 RAG 检索链路溯源 (白盒透视)", open=False):
                trace_output = gr.HTML("<div style='color:gray; padding:10px;'>等待首次检索...</div>")

    # ================= 绑定事件驱动逻辑 =================
    
    # 1. 提交问题：先清空输入框并把问题放到聊天界面的用户侧，然后触发 Bot 处理
    submit_event = msg_input.submit(
        user_input, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot], queue=False
    ).then(
        bot_response, 
        inputs=[chatbot, slider_top_k, slider_top_n, slider_threshold], # 👈 加了 slider_threshold
        outputs=[chatbot, trace_output]
    )
    
    submit_btn.click(
        user_input, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot], queue=False
    ).then(
        bot_response, 
        inputs=[chatbot, slider_top_k, slider_top_n, slider_threshold], # 👈 加了 slider_threshold
        outputs=[chatbot, trace_output]
    )
    
    # 2. 清空聊天
    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=custom_css, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"))