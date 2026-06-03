import json
import re
import sys
import os

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

    # 策略3：选项文本内容匹配——模型可能直接输出了选项内容而非字母
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
    因果推理评测：多策略提取 + 低置信度时二次确认
    """
    print(f"因果关系--处理第{item['rowIdx']}条数据")

    options_str = "，".join(item['options'])
    add_prompt = (
        f"请分析以下因果关系问题，并从选项中选择最合理的答案。\n"
        f"问题：{item['question']}\n"
        f"选项：{options_str}\n"
        f"要求：请先简要说明推理过程，最后务必以'【答案】选项字母'的形式结束（例如：【答案】A）。"
    )

    try:
        answer = call_api(model, add_prompt)
        pred, confidence = extract_choice(answer, item['options'])

        # 低置信度或提取失败时，用二次确认 prompt 再问一次
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

        score = 1 if pred == item['answer'] else 0
        print(f"模型回答提取: {pred} (置信度: {confidence}), 正确答案: {item['answer']}, 得分: {score}")
        return answer, score
    except Exception as e:
        print(f"因果推理 API 调用异常: {e}")
        return "ERROR", 0


def evaluate(model, qs_list):
    if not qs_list:
        return [], 0.0

    result = []
    response_ls = []
    for item in qs_list:
        response, score = deal(model, item)
        result.append(score)

        # 清理响应中的多余空行，保持结果整洁
        clean_response = re.sub(r"\n\s*\n", "\n", response)

        response_ls.append({
            "dataId": item['rowIdx'],
            "response": clean_response,
            "is_correct": bool(score)
        })

    # 计算百分制得分
    final_score = (sum(result) / len(result)) * 100 if result else 0.0
    return response_ls, round(final_score, 2)



if __name__ == "__main__":
    # for model in ['yi-lightning',"ernie-4.0-8k"]:
    #     print("正在执行{}模型".format(model))
    #     try:
    #         result=  evaluate(model)
    #         print(result)
    #     except:
    #         print("模型{}执行错误".format(model))
    #         continue
    result=  evaluate("gpt-4o-mini")
    print(result)