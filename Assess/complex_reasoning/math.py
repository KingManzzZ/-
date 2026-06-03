import os
import re
import sys

# 统一导入处理
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import call_api

# 裁判模型，用于评估推理过程的逻辑正确性
JUDGE_MODEL = "Qwen-max"


def normalize_number(text):
    """
    增强数值提取和标准化：支持分数、百分比、科学计数法等
    返回 float 或 None
    """
    if not text:
        return None
    text = text.strip().replace(",", "").replace(" ", "")

    # 百分比
    pct_match = re.match(r"^([-+]?\d*\.?\d+)\s*%$", text)
    if pct_match:
        return float(pct_match.group(1)) / 100.0

    # 分数
    frac_match = re.match(r"^([-+]?\d+)\s*/\s*(\d+)$", text)
    if frac_match:
        denom = float(frac_match.group(2))
        if denom != 0:
            return float(frac_match.group(1)) / denom
        return None

    # 科学计数法或普通数字
    try:
        return float(text)
    except ValueError:
        return None


def numbers_equal(num_str1, num_str2, tolerance=1e-6):
    """
    判断两个数值字符串是否在容差范围内相等
    """
    val1 = normalize_number(num_str1)
    val2 = normalize_number(num_str2)

    if val1 is None or val2 is None:
        return str(num_str1).strip() == str(num_str2).strip()

    if val2 == 0:
        return abs(val1) < tolerance

    return abs(val1 - val2) / max(abs(val2), 1e-10) < tolerance


def judge_reasoning_process(pred_process, ref_process, question, judge_model=JUDGE_MODEL):
    """
    使用裁判模型评估推理过程的逻辑正确性，返回 0-1 的分数
    """
    if not ref_process or not ref_process.strip():
        return None  # 无参考过程时返回 None，由调用方决定如何处理

    judge_prompt = (
        f"你是一个数学推理评测专家。请评估[模型解题过程]的逻辑正确性。\n\n"
        f"[题目]: {question}\n"
        f"[参考解题过程]: {ref_process}\n"
        f"[模型解题过程]: {pred_process}\n\n"
        f"评分标准：\n"
        f"- 1.0：推理逻辑完全正确，步骤清晰合理\n"
        f"- 0.7-0.9：推理方向正确，有小瑕疵但不影响逻辑链\n"
        f"- 0.4-0.6：部分推理正确，但存在逻辑漏洞或方法不当\n"
        f"- 0.1-0.3：推理方向基本错误，仅有零星正确步骤\n"
        f"- 0.0：完全错误或无关内容\n\n"
        f"注意：不同的解法思路只要逻辑正确都应给高分，不要求和参考过程完全一致。\n"
        f"请严格按照以下格式返回：【分数】你的评分数字"
    )

    try:
        response = call_api(judge_model, judge_prompt, temperature=0.1)
        match = re.search(r"【分数】\s*([01](?:\.\d+)?)", response)
        if match:
            return float(match.group(1))

        # 备用匹配：直接找 0-1 之间的数字
        fallback = re.findall(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", response)
        if fallback:
            return float(fallback[-1])
    except Exception as e:
        print(f"裁判模型调用失败: {e}")

    return None  # 裁判失败时返回 None


def deal(model, item):
    """
    数学推理评测：LLM-as-Judge 评估过程 + 增强数值匹配 + 合理计分公式
    """
    try:
        r_parts = item['answer'].split("#### ")
        r_process = r_parts[0].strip()
        r_answer = r_parts[1].strip()
    except (IndexError, AttributeError):
        r_process = ""
        r_answer = str(item['answer']).strip()

    add_prompt = (
        f"请逐步解答以下数学题。确保最后一行以'#### 数字'的格式给出最终数值答案。\n"
        f"题目：{item['question']}\n"
        f"输出要求：\n1. 请先提供清晰的解题步骤；\n2. 最后以格式'#### 最终答案数字'结尾。"
    )

    try:
        answer = call_api(model, add_prompt)

        # --- 提取预测结果与过程 ---
        process = answer
        num = ""
        if "####" in answer:
            parts = answer.split("####")
            process = parts[0].strip()
            num_match = re.search(r"[-+]?\d[\d,]*\.?\d*(?:/\d+)?(?:%)?", parts[-1])
            num = num_match.group().replace(",", "") if num_match else ""
        else:
            all_nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", answer)
            num = all_nums[-1].replace(",", "") if all_nums else ""

        # --- 数值匹配判定 ---
        is_match = numbers_equal(num, r_answer)

        # --- 推理过程评估 ---
        process_score = judge_reasoning_process(process, r_process, item['question'])

        # --- 计分模型（百分制） ---
        # 答案正确 + 过程优秀：满分
        # 答案正确 + 无过程评估：给予较高基础分（70）
        # 答案错误 + 过程优秀：给予中等分数（最高50）
        # 答案错误 + 过程差：低分
        if is_match:
            if process_score is not None:
                # 答案对：70基础分 + 30分过程加成
                score = 70 + (30 * process_score)
            else:
                # 无法评估过程但答案对：给 75 分（不满分，避免蒙对高分）
                score = 75.0
        else:
            if process_score is not None:
                # 答案错但过程有价值：最高 50 分
                score = 50 * process_score
            else:
                score = 0.0

        print(f"[数学评测] 模型预测: {num}, 参考: {r_answer} | "
              f"数值匹配: {is_match} | 过程评分: {process_score} | 得分: {score:.2f}")
        return answer, score

    except Exception as e:
        print(f"数学评测运行时异常: {e}")
        return f"RUNTIME_ERROR: {str(e)}", 0


def evaluate(model, qs_list):
    if not qs_list:
        return [], 0.0

    result = []
    response_ls = []
    for item in qs_list:
        print(f"数学--处理第{item['rowIdx']}条数据")
        response, score = deal(model, item)

        result.append(score)
        response_ls.append({
            "dataId": item['rowIdx'],
            "response": re.sub(r"\n\s*\n", "\n", response),
            "score": round(score, 2)
        })

    final_score = (sum(result) / len(result)) if result else 0.0
    return response_ls, round(final_score, 2)

if __name__ == "__main__":
    ls=[]
    for model in ["qwen-max"]:
            result=  evaluate(model)
            print(result)
            ls.append(result)
    print(ls)