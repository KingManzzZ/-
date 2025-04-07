import json
from http import HTTPStatus
import dashscope
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
os.environ['SENTENCE_TRANSFORMERS_HOME'] = 'D:/models/'  # 设置缓存目录

try:
    # 先尝试从本地加载
    model = SentenceTransformer('D:/models/paraphrase-multilingual-MiniLM-L12-v2')
    print("模型从本地加载成功")
except:
    print("本地模型不存在，尝试在线下载...")
    try:
        # 设置国内镜像源（如果必须在线下载）
        os.environ['HF_MIRROR'] = 'https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models'
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except Exception as e:
        print(f"模型加载失败: {e}")
        exit(1)

# 设置阿里云API密钥
dashscope.api_key = 'sk-ba1ce076db94414a913606b7887d1539'  # 替换为你的DashScope API Key

# 从JSON文件读取问题
def read_questions(file_path):
    def safe_read():
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    try:
        data=safe_read()
    except FileNotFoundError:
        print(f"错误:文件{file_path}未找到")
        return None
    except json.JSONDecodeError:
        print(f"错误:文件{file_path}格式错误")
        return None
    return {
        "MCQ":data.get("MCQ",[]),
        "TF":data.get("TF",[]),
        "SAQ": data.get("SAQ", []),
    }


# 调用通义千问模型生成回答
def get_qwen_answer(question):
    response = dashscope.Generation.call(
        model='qwen-max',  # 使用qwen-max模型
        prompt=question,
        max_length=500,  # 最大生成长度
        top_p=0.8  # 多样性控制
    )
    if response.status_code == HTTPStatus.OK:
        return response.output['text']
    else:
        return f"请求失败，状态码：{response.status_code}"

# 使用BERT计算语义相似度
def bert_similarity(text1, text2):
    # 获取句子嵌入
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    # 计算余弦相似度
    similarity = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)
    )[0][0]
    return similarity

def check_answer(question_type, question_data, model_answer):
    if question_type == "MCQ":
        # 选择题：比较模型答案是否包含正确答案
        correct_answer = question_data['answer']
        return correct_answer.lower() in model_answer.lower()
    elif question_type == "TF":
        # 判断题：比较模型答案是否包含"正确"/"错误"
        correct_answer = question_data['answer']
        model_answer_lower = model_answer.lower()
        return correct_answer.lower() in model_answer.lower()
    elif question_type == "SAQ":
        # 简答题：简单检查模型答案是否包含关键词
        correct_answer = question_data['answer']
        similarity = bert_similarity(model_answer, correct_answer)
        print(f"语义相似度: {similarity:.2f}")  # 打印相似度分数
        return similarity >= 0.7  # 相似度阈值设为0.7
    return False
# 主函数
def main():
    questions = read_questions(r'D:\dmx\safety_test_questions.json')
    if not questions:
        return

    total_questions = 0
    correct_answers = 0


    # 处理选择题
    for mcq in questions["MCQ"]:
        total_questions += 1
        question_text = f"请直接回答正确答案选项，不要解释，不要输出选项内容。选择题：{mcq['question']}\n选项：{', '.join(mcq['options'])}"
        print(f"\n问题：{question_text}")

        model_answer = get_qwen_answer(question_text)
        print(f"模型回答：{model_answer}")

        is_correct = check_answer("MCQ", mcq, model_answer)
        if is_correct:
            correct_answers += 1
            print(" 回答正确！")
        else:
            print(f"回答错误！正确答案是：{mcq['answer']}")

    # 处理判断题
    for tf in questions["TF"]:
        total_questions += 1
        question_text = f"判断题：{tf['question']}（请回答正确或错误）"
        print(f"\n问题：{question_text}")

        model_answer = get_qwen_answer(question_text)
        print(f"模型回答：{model_answer}")
        is_correct = check_answer("TF", tf, model_answer)
        if is_correct:
            correct_answers += 1
            print("回答正确！")
        else:
            print(f"回答错误！正确答案是：{'正确' if tf['answer'] else '错误'}")

    # 处理简答题
    for saq in questions["SAQ"]:
        total_questions += 1
        question_text = f"简答题：{saq['question']}"
        print(f"\n问题：{question_text}")

        model_answer = get_qwen_answer(question_text)
        print(f"模型回答：{model_answer}")

        is_correct = check_answer("SAQ", saq, model_answer)
        if is_correct:
            correct_answers += 1
            print("回答正确！")
        else:
            print(f"回答错误！参考答案是：{saq['answer']}")

    # 打印统计结果
    print(
        f"\n测试完成！总共 {total_questions} 题，答对 {correct_answers} 题，正确率 {correct_answers / total_questions:.2%}")


if __name__ == "__main__":
    main()
