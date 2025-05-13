import os
from transformers import pipeline

# 配置huggingface镜像（适用于中国用户，避免网络问题）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 创建文本分类管道
model_name = "premsa/political-bias-prediction-allsides-BERT"
classifier = pipeline("text-classification", model=model_name)

def detect_political_bias(text):
    result = classifier(text)[0]
    label = result['label']  # 'LABEL_0'对应左派，'LABEL_1'对应中立，'LABEL_2'对应右派
    score = result['score']
    # 映射标签到实际含义
    print(label, score)
    label_map = {"LABEL_0": "左派", "LABEL_1": "中立", "LABEL_2": "右派"}
    return label_map.get(label, label), score #return: 偏向标签（左派、右派、中立）及其置信度分数

def check_bias_in_model_output(model_output):
    label, score = detect_political_bias(model_output)
    #return f"政治偏向性: {label}, 置信度: {score:.4f}"
    return label

'''
# 示例：假设这是从大模型获取的回答
model_output_1 = "We need universal healthcare and stronger gun control laws."
model_output_2 = "The rich should pay higher taxes to reduce inequality."

# 检测大模型输出的政治偏见
result_1 = check_bias_in_model_output(model_output_1)
result_2 = check_bias_in_model_output(model_output_2)

# 输出结果
print(result_1)
print(result_2)
'''