# DeepSeek API
from typing import List
from openai import OpenAI

class LLMGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
    def generate_multi_turn_stream(self, query: str, context_docs: List[str], history: List):
        """
        支持多轮对话与流式输出打字机效果
        """
        context_str = "\n\n---\n\n".join(context_docs)
        
        # 1. 设定系统角色
        messages = [{"role": "system", "content": "你是国内顶尖的三甲医院资深临床专家。假如问的是医学方面的问题,请严谨作答，推荐药物必须基于参考资料。"}]
        
        # 2. 注入历史对话记录，让模型有“记忆”
        for msg in history:
            # 兼容 Gradio 6.0 新版格式：字典类型 {"role": "user", "content": "..."}
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
                
            # 兼容老版 Gradio 格式：列表或元组类型 [user_msg, bot_msg]
            elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                if msg[0]: messages.append({"role": "user", "content": msg[0]})
                if msg[1]: messages.append({"role": "assistant", "content": msg[1]})
            
        # 3. 构造当前轮次的 Prompt（包含知识库内容）
        current_prompt = f"""请基于以下[知识库上下文]解答用户最新问题。
        要求：
        1. 【医疗专业问题】：若问题涉及疾病、症状、药物、诊断等，必须严格参考上下文。若上下文有直接答案，请提炼作答；若上下文提供的是通用原则，请合理推断。只有在上下文中完全找不到任何相关线索时，才回答“知识库未找到相关方案”，严禁编造医疗建议。
        2. 【部分命中处理（极其重要）】：如果上下文只能回答用户的“部分问题”（例如找到了推荐药物，但没写怎么备药），请直接基于已有信息作答，并结合医学常识给出合理建议（如：处方药不建议自行备药）。在这种情况下，【绝对不要】输出“知识库未找到相关方案”这几个字！
        3. 【日常对话问题】：若用户只是进行日常打招呼、表达感谢、询问你的身份/能力等（非医疗业务问题），请直接以“医疗AI智能助理”的身份用专业、友好的口吻自然回复，无需参考知识库。

[知识库上下文]:
{context_str}

[用户最新问题]: {query}
"""
        messages.append({"role": "user", "content": current_prompt})
        
        # 4. 流式请求大模型
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.1,
            stream=True # 开启流式输出
        )
        
        # 5. 实现打字机生成器
        partial_message = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                partial_message += chunk.choices[0].delta.content
                yield partial_message



    def rewrite_query(self, query: str, history: List) -> str:
        """
        核心亮点：利用 LLM 进行提取会话的历史记录形成摘要，在回答问题前查看这个摘要防止错过上下文信息
        """
        if not history:
            return query
            
        # 1. 格式化整个对话历史
        history_text = ""
        for msg in history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                role = "用户" if msg["role"] == "user" else "AI"
                history_text += f"{role}: {msg['content']}\n"
            elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                if msg[0]: history_text += f"用户: {msg[0]}\n"
                if msg[1]: history_text += f"AI: {msg[1]}\n"

        # 2. 编写重写 Prompt（完全按照你的思路实现）
        prompt = f"""你是一个医疗系统的底层搜索词重写引擎。请阅读[对话历史]和[用户最新问题]，将其重写为一个独立、完整、毫无歧义的搜索词。
       【核心重写规则】：
        1. 全局视野：结合完整的对话历史，理解用户的提问语境。
        2. 话题切换（极其重要）：如果用户的最新问题提出了一个【全新的疾病或话题】，请立刻彻底抛弃历史记录中的旧疾病，绝对不要把旧疾病和新疾病拼在一起！
        3. 指代消解：如果最新问题使用了代词（如“这病怎么治”、“推荐什么药”），必须从最后一次AI回复中提取核心疾病名称进行替换。

       【参考示例】：
        示例1（指代消解）：
        历史：用户: 介绍一下百日咳。 AI: 百日咳是...
        最新问题：这病大概多久能好？
        输出：百日咳 恢复时间

       示例2（话题切换 - 必须抛弃旧话题）：
       历史：用户: 范科尼综合征是什么？ AI: 范科尼综合征是...
       最新问题：那么肌纤维组织炎怎么预防呢？
       输出：肌纤维组织炎 预防方法

[对话历史]：
{history_text}

[用户最新问题]：{query}

请严格根据规则和示例，直接输出重写后的“搜索词”（绝不包含任何多余标点或语气词）："""

        # 3. 调用大模型进行秒级推理
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0 # 严谨任务，温度设为0
        )
        
        rewritten_query = response.choices[0].message.content.strip()
        return rewritten_query            