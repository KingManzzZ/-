import time
import json
import os
import sys
import re

# 统一导入处理
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import call_api


def split_text_into_chunks(text, max_chars=3000):
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    current_chunk = ""

    # 按段落分割（以换行符为分隔符）
    paragraphs = text.split('\n')
    
    for paragraph in paragraphs:
        # 如果当前段落加入后不超过限制，则加入
        if len(current_chunk) + len(paragraph) + 1 <= max_chars:
            if current_chunk:
                current_chunk += '\n' + paragraph
            else:
                current_chunk = paragraph
        else:
            # 如果当前段落本身超过限制，需要进一步分割
            if len(paragraph) > max_chars:
                # 保存当前块
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # 按句子分割长段落
                sentences = paragraph.split('。')
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= max_chars:
                        if current_chunk:
                            current_chunk += '。' + sentence
                        else:
                            current_chunk = sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
            else:
                # 保存当前块，开始新块
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def send_long_text_as_conversation(model, long_text, question, max_chars=3000):
    """将长文本分块发送给模型，使用系统级消息堆叠 + 中间轮次健康检查"""
    # 切割文本
    text_chunks = split_text_into_chunks(long_text, max_chars)
    print(f"长文本被切割为 {len(text_chunks)} 个块")
    
    # 构造消息序列
    # 初始系统消息，明确告知后续是长文本输入
    messages = [
        {"role": "system", "content": "你是一个拥有超强记忆力的助手。接下来的消息将分段发送一段长文本，请你保持极简回复（仅回复OK），直到最后接收到具体的提问后，再根据全文内容进行详细回答。"}
    ]

    # 逐块发送，增加健康检查
    failed_chunks = 0
    for idx, chunk in enumerate(text_chunks):
        content = f"[长文本数据 {idx + 1}/{len(text_chunks)}]:\n\n{chunk}"
        messages.append({"role": "user", "content": content})

        # 实时发送并获取简单的确认，确保模型“看到”了这部分数据
        response = call_api(model, None, messages=messages)
        response_stripped = response.strip() if response else ""
        print(f"第 {idx + 1} 块发送完成，模型状态: {response_stripped[:10]}")

        # 健康检查：空响应或错误标识时记录
        if not response_stripped or "error" in response_stripped.lower():
            failed_chunks += 1
            print(f"  ⚠️ 第 {idx + 1} 块响应异常，已累计 {failed_chunks} 次失败")
            if failed_chunks >= 3:
                print("  ❌ 连续失败过多，终止分块发送")
                return None
            messages.append({"role": "assistant", "content": "OK"})
        else:
            failed_chunks = 0
            messages.append({"role": "assistant", "content": response})

        time.sleep(0.5)

    # 最终提问
    final_query = f"以上是全部参考文本。现在请回答问题：{question}"
    messages.append({"role": "user", "content": final_query})

    final_answer = call_api(model, None, messages=messages)
    return final_answer


JUDGE_MODEL = "Qwen-max"
FUZZY_LOW = 0.3
FUZZY_HIGH = 0.7


def tokenize_chinese(text):
    """简易中英文分词：中文按字切分，英文按空格/标点切分，统一小写"""
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
    """计算 token 级 F1-score"""
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
    """LLM Judge：仅在 F1 落在模糊区间时触发"""
    judge_prompt = (
        f"你是一个阅读理解评测专家。请判定[模型回答]是否正确回答了[问题]。\n\n"
        f"[问题]: {question}\n"
        f"[参考答案]: {ref}\n"
        f"[模型回答]: {pred}\n\n"
        f"评分标准：\n"
        f"- 1.0：语义完全正确\n"
        f"- 0.7-0.9：核心信息正确，有少量遗漏\n"
        f"- 0.4-0.6：部分正确，缺少关键信息\n"
        f"- 0.1-0.3：大部分错误\n"
        f"- 0.0：完全错误\n\n"
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
    """处理单个测试项：分块发送 + F1-score + 模糊区间触发 LLM Judge"""
    print("----- [记忆能力] 处理第{}条数据 -----".format(item['rowIdx'] + 1))

    long_text = item.get('context', '') or item.get('text', '') or item.get('question', '')
    if isinstance(long_text, dict):
        long_text = long_text.get('content', '') or long_text.get('text', '')

    question = item.get('query', '') or item.get('question', '')
    if isinstance(question, dict):
        question = question.get('query', '') or question.get('content', '')

    if not long_text or not question:
        print("警告：内容缺失")
        return "ERROR", 0.0

    try:
        answer = send_long_text_as_conversation(model, long_text, question)

        # 健康检查：分块发送失败时返回 None
        if answer is None:
            print("分块发送失败，跳过此题")
            return "ERROR", 0.0

        print(f"模型回答摘要: {answer[:100]}...")
        standard_answer = str(item.get('answer', ''))

        # L1：精确/子串匹配
        if standard_answer in answer or answer in standard_answer:
            return answer, 1.0

        # L2：F1-score
        f1 = compute_f1(answer, standard_answer)

        # L3：模糊区间触发 LLM Judge
        if FUZZY_LOW < f1 < FUZZY_HIGH:
            judge_result = llm_judge_score(answer, standard_answer, question)
            if judge_result is not None:
                f1 = judge_result
                print(f"  → LLM Judge 触发，裁判评分: {judge_result}")

        print(f"  F1 得分: {f1:.3f}")
        return answer, f1
    except Exception as e:
        print(f"处理失败: {e}")
        return "ERROR", 0.0


def evaluate(model, qs_list=None):
    """评估模型的记忆能力"""
    if qs_list is None:
        # 从文件读取数据
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        direction = os.path.join(project_root, "dataset", "performance", "长文本理解能力", "记忆能力", "memory_retention.json")
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
                "response": re.sub(r"\n\s*\n", "\n", str(response)),
                "score": round(score, 2)
            })
    except Exception as e:
        print(f"记忆评测中断: {e}")

    avg_score = (sum(result) / len(result)) * 100 if result else 0.0
    return response_ls, round(avg_score, 2)


if __name__ == "__main__":
    ls = []
    for model in ["DeepSeek-V3"]:
        print("正在执行{}模型".format(model))
        result = evaluate(model)
        print(result)
        ls.append(result)
    print(ls)
