import os
import re
from typing import List

class DocumentLoader:
    def load_directory(self, dir_path: str) -> List[str]:
        texts = []
        for filename in os.listdir(dir_path):
            if filename.endswith(".txt"):
                with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
        return texts

class TextCleaner:
    def clean(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

class TextChunker:
    def __init__(self, chunk_size: int = 400):
        self.chunk_size = chunk_size
        # 优先级：双换行 > 单换行 > 句号/感叹号 > 逗号
        self.separators = ["\n\n", "\n", "。", "！", "，"]

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """纯净手写的递归字符切分算法"""
        if len(text) <= self.chunk_size:
            return [text]
            
        # 找到当前文本中存在的最高优先级分隔符
        sep = ""
        for s in separators:
            if s in text:
                sep = s
                break
                
        # 如果连逗号都找不到，只能按字硬切
        if not sep:
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
            
        splits = text.split(sep)
        chunks, current = [], ""
        
        for i, s in enumerate(splits):
            part = s + sep if i < len(splits) - 1 else s
            if len(current) + len(part) <= self.chunk_size:
                current += part
            else:
                if current: chunks.append(current.strip())
                current = part
        if current: chunks.append(current.strip())
            
        # 如果切出来的块依然超过大小限制，继续用更低一级的分隔符递归切分
        final_chunks = []
        next_seps = separators[separators.index(sep)+1:] if sep in separators else []
        for c in chunks:
            if len(c) > self.chunk_size and next_seps:
                final_chunks.extend(self._recursive_split(c, next_seps))
            else:
                final_chunks.append(c)
                
        return final_chunks

    def split(self, text: str) -> List[str]:
        # 【面试亮点：上下文感知切分】优先级大于递归字符切分
        # 1. 先按疾病名称把不同疾病分开，只有当‘名称: ’这三个字紧紧贴在某一行的最开头时，才切分它
        entities = re.split(r'(?m)^(?=名称: )', text)
        all_chunks = []
        
        for entity_text in entities:
            if not entity_text.strip(): continue
            
            # 2. 提取当前疾病的名字
            name_match = re.search(r'名称:\s*(.+)', entity_text)
            disease_name = name_match.group(1).strip() if name_match else "未知疾病"
            
            # 3. 对该疾病的内容进行递归细粒度切分
            sub_chunks = self._recursive_split(entity_text.strip(), self.separators)
            
            # 4. 把疾病名称和绝对序号作为 Metadata 缝合进每一个 Chunk！
            for idx,chunk in enumerate(sub_chunks):
                # 不管是不是头块，无差别注入 Metadata 前缀！
                chunk = f"【关联疾病: {disease_name} | 序号: {idx}】\n{chunk}"
                all_chunks.append(chunk)
                
        return all_chunks