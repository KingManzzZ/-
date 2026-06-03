import json
import time
import os
import re
import sys

# 统一导入处理
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import call_api

def extract_choice(answer_text, options):
    """
    多策略选项提取：正则格式 > 选项文本匹配 > 最后字母兜底
    返回提取到的字母和置信度（high/medium/low）
    """
    # 策略1：正则提取【答案】标识
    match = re.search(r"【答案】\s*([A-D])", answer_text)
    if match:
        return match.group(1), "high"

    # 策略2：匹配 "答案是X"、"选X"、"应该选X" 等常见表述
    pattern = re.search(r"(?:答案|选择|应选|选)\s*(?:是|为)?\s*([A-D])", answer_text)
    if pattern:
        return pattern.group(1), "high"

    # 策略3：选项文本内容匹配
    option_letters = ['A', 'B', 'C', 'D']
    for idx, opt_text in enumerate(options):
        if idx < len(option_letters):
            clean_opt = opt_text.strip().lstrip("ABCD.、:： ")
            if clean_opt and clean_opt in answer_text:
                return option_letters[idx], "medium"

    # 策略4：取最后出现的选项字母（低置信度兜底）
    clean_answer = re.sub(r'[^A-D]', '', answer_text.upper())
    if clean_answer:
        return clean_answer[-1], "low"

    return "", "none"


def deal(model, item):
    """
    常识逻辑推理评测：多策略提取 + 低置信度时二次确认
    """
    print(f"常识逻辑--处理第{item['rowIdx']}条数据")

    options_str = "，".join(item['options']) if isinstance(item['options'], list) else str(item['options'])
    add_prompt = (
        f"请基于常识和基本逻辑解答以下问题。\n"
        f"问题：{item['question']}\n"
        f"选项：{options_str}\n"
        f"请在推理后直接给出选项字母，并务必在最后一行以'【答案】选项字母'的形式结束（例如：【答案】B）。"
    )

    try:
        answer = call_api(model, add_prompt)
        print(f"原始响应摘要: {answer[:50]}...")

        options_list = item['options'] if isinstance(item['options'], list) else []
        pred, confidence = extract_choice(answer, options_list)

        # 低置信度或提取失败时，用二次确认 prompt
        if confidence in ("low", "none"):
            confirm_prompt = (
                f"你刚才对以下问题进行了分析：\n{item['question']}\n"
                f"选项：{options_str}\n"
                f"请直接告诉我你的最终答案字母（A/B/C/D），只输出一个字母，不要有其他内容。"
            )
            confirm_answer = call_api(model, confirm_prompt, temperature=0.1)
            confirm_match = re.search(r"([A-D])", confirm_answer.strip().upper())
            if confirm_match:
                pred = confirm_match.group(1)
                confidence = "confirmed"

        is_correct = 1 if pred == item['answer'] else 0
        print(f"回答是{pred} (置信度: {confidence}), 答案是{item['answer']}, 判定: {'通过' if is_correct else '失败'}")
        return answer, is_correct
    except Exception as e:
        print(f"逻辑推理 API 调用异常: {e}")
        return "ERROR", 0


def evaluate(model, qs_list):
    if not qs_list:
        return [], 0.0

    result = []
    response_ls = []
    for item in qs_list:
        response, score = deal(model, item)
        result.append(score)

        response_dic = {
            "dataId": item['rowIdx'],
            "response": re.sub(r"\n\s*\n", "\n", response),
            "is_correct": bool(score)
        }
        response_ls.append(response_dic)

    final_score = (sum(result) / len(result)) * 100 if result else 0.0
    print(f"逻辑评估完成: 已处理 {len(result)} 条数据，平均得分: {final_score}")
    return response_ls, round(final_score, 2)

if __name__ == "__main__":
    ls=[]
    for model in ['DeepSeek-V3',"qwen-max"]:
        print("正在执行{}模型".format(model))
        result=  evaluate(model)
        print(result)
        ls.append(result)
    print(ls)
