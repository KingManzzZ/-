import json
import time
import os
import sys
import re

# 统一导入处理
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import call_api

# LLM Judge 仅在模糊区间触发
JUDGE_MODEL = "Qwen-max"
FUZZY_LOW = 0.3
FUZZY_HIGH = 0.7


def tokenize_chinese(text):
    """
    简易中英文分词：中文按字切分，英文按空格/标点切分，统一小写
    """
    tokens = []
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            tokens.append(char)
        elif char.isalnum():
            if tokens and tokens[-1].isalnum():
                tokens[-1] += char
            else:
                tokens.append(char)
    return [t.lower() for t in tokens if t.strip()]


def compute_f1(prediction, reference):
    """
    计算 token 级 F1-score（SQuAD 标准做法）
    """
    pred_tokens = tokenize_chinese(prediction)
    ref_tokens = tokenize_chinese(reference)

    if not pred_tokens or not ref_tokens:
        return 1.0 if pred_tokens == ref_tokens else 0.0

    common = set(pred_tokens) & set(ref_tokens)
    num_common = sum(min(pred_tokens.count(t), ref_tokens.count(t)) for t in common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def llm_judge_score(pred, ref, question, judge_model=JUDGE_MODEL):
    """
    LLM Judge：仅在 F1 落在模糊区间时触发
    """
    judge_prompt = (
        f"你是一个信息提取评测专家。请判定[模型提取结果]是否准确回答了[问题]。\n\n"
        f"[问题]: {question}\n"
        f"[参考答案]: {ref}\n"
        f"[模型提取结果]: {pred}\n\n"
        f"评分标准：\n"
        f"- 1.0：提取的信息完全准确，与参考答案语义一致\n"
        f"- 0.7-0.9：核心信息正确，有少量遗漏或多余内容\n"
        f"- 0.4-0.6：部分正确，关键信息有缺失\n"
        f"- 0.1-0.3：大部分错误，仅有零星正确信息\n"
        f"- 0.0：完全错误或无关\n\n"
        f"请严格按照以下格式返回：【分数】你的评分数字"
    )
    try:
        response = call_api(judge_model, judge_prompt, temperature=0.1)
        match = re.search(r"【分数】\s*([01](?:\.\d+)?|0?\.\d+)", response)
        if match:
            return float(match.group(1))
        fallback = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", response)
        if fallback:
            return float(fallback[-1])
    except Exception as e:
        print(f"裁判模型调用失败: {e}")
    return None


def deal(model, item):
    """
    信息提取评测：长文本传入 + F1-score 对提取结果评分 + 模糊区间触发 LLM Judge
    """
    print(f"----- [信息提取] 处理第{item['rowIdx'] + 1}条数据 -----")

    # 获取长文本上下文
    context = item.get('context', '') or item.get('text', '') or ''
    if isinstance(context, dict):
        context = context.get('content', '') or context.get('text', '')

    question = item.get('question', '') or item.get('query', '')
    if isinstance(question, dict):
        question = question.get('content', '') or question.get('query', '')

    # 构建 prompt，确保长文本被传入
    prompt = (
        f"请在以下提供的长文本中定位并提取问题的答案。\n\n"
        f"【长文本】\n{context}\n\n"
        f"【问题】{question}\n\n"
        f"注意：仅回答提取到的具体事实，不要包含无用的解释。\n"
        f"要求：首先输出你的提取逻辑或原文出处，最后以'【答案】提取内容'的形式结束。"
    )

    try:
        answer = call_api(model, prompt)
        print(f"提取结果: {answer[:100]}...")

        standard_answer = str(item.get('answer', ''))

        # 提取【答案】后的内容作为预测（仅对 pred 做评分，不对整个 response）
        match = re.search(r"【答案】\s*(.+)", answer, re.DOTALL)
        pred = match.group(1).strip().split('\n')[0] if match else answer

        # L1：精确匹配（含子串包含）
        if standard_answer.strip() == pred.strip():
            return answer, 1.0
        if standard_answer in pred or pred in standard_answer:
            return answer, 1.0

        # L2：F1-score（对 pred 而非全文）
        f1 = compute_f1(pred, standard_answer)

        # L3：模糊区间触发 LLM Judge
        if FUZZY_LOW < f1 < FUZZY_HIGH:
            judge_result = llm_judge_score(pred, standard_answer, question)
            if judge_result is not None:
                f1 = judge_result
                print(f"  → LLM Judge 触发，裁判评分: {judge_result}")

        print(f"  F1 得分: {f1:.3f}")
        return answer, f1
    except Exception as e:
        print(f"信息提取请求失败: {e}")
        return "ERROR", 0.0


def evaluate(model, qs_list=None):
    """
    信息提取指标评测方法。
    """
    if qs_list is None:
        # 保持对原有文件读取方式的兼容
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        direction = os.path.join(project_root, "dataset", "performance", "长文本理解能力", "信息提取", "information_extraction.json")
        with open(direction, 'r', encoding='utf-8') as f:
            qs_list = json.load(f)

    result = []
    response_ls = []

    try:
        for item in qs_list:
            response, score = deal(model, item)
            result.append(score)

            response_ls.append({
                "dataId": item.get('rowIdx', 0),
                "response": re.sub(r"\n\s*\n", "\n", response),
                "score": round(score, 2)
            })
    except Exception as e:
        print(f"评测中断: {e}")

    avg_score = (sum(result) / len(result)) * 100 if result else 0.0
    return response_ls, round(avg_score, 2)
if __name__ == "__main__":
    ls=[]
    for model in ['DeepSeek-V3',"qwen-max","yi-lightning"]:
        print("正在执行{}模型".format(model))
        result=  evaluate(model)
        print(result)
        ls.append(result)
    print(ls)


# def locate_answer_paragraphs(item, labels):
#     """根据labels中的字符位置定位答案所在段落"""
#     paragraphs = item["question"]["content"]
#     answer_pars = set()
#
#     # 计算每个段落的字符偏移量
#     char_offset = 0
#     for idx, par in enumerate(paragraphs):
#         par_length = len(par) + 4  # 包括分隔符占位长度
#         for label in labels:
#             start, end = label["start"][0], label["end"][0]  # 取第一个位置范围
#             if char_offset <= start < char_offset + par_length:
#                 answer_pars.add(idx + 1)  # 段落编号从1开始
#         char_offset += par_length
#     return sorted(list(answer_pars))  # 返回有序段落编号列表