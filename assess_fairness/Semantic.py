import os
import torch
from transformers import AutoTokenizer, AutoModel
import re
import jieba

# 国内镜像配置（必须放在最顶部）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 初始化分词器
jieba.initialize()
# 初始化 tokenizer 和 model
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def preprocess(text):
    """中文文本预处理"""
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)  # 保留中文和基本字符
    return " ".join(jieba.cut(text))

def get_embedding(text):
    """获取文本向量"""
    inputs = tokenizer(
        preprocess(text),
        return_tensors="pt",
        max_length=64,
        truncation=True,
        padding='max_length'
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # 取[CLS]向量

def compare_texts(text1, text2):
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    # 计算余弦相似度
    sim = torch.cosine_similarity(embedding1, embedding2).item()
    return round(sim, 2)