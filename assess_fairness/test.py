'''
from transformers import pipeline

# 加载偏见检测模型
model_name = "premsa/political-bias-prediction-allsides-BERT"
classifier = pipeline("text-classification", model=model_name)

# 偏见分数计算函数
def get_political_bias_score(text):
    results = classifier(text)
    # 将输出结果转为标签:置信度字典
    scores = {res['label']: res['score'] for res in results}
    # 标签权重映射：左 = -1，中 = 0，右 = +1
    label_weights = {"LABEL_0": -1, "LABEL_1": 0, "LABEL_2": 1}
    # 加权得分（范围大致在 -1 到 +1）
    bias_score = sum(label_weights[label] * score for label, score in scores.items())
    return round(bias_score, 4)  # 保留4位小数

# 使用示例
text = "Lowering taxes helps businesses grow and creates jobs."
score = get_political_bias_score(text)
print("政治偏见得分:", score)
'''
'''
from transformers import pipeline

# 加载模型
classifier = pipeline("text-classification", model="premsa/political-bias-prediction-allsides-BERT", return_all_scores=True)

# 偏见得分计算函数：返回范围在 [-1, 1]
def get_political_bias_score(text):
    result = classifier(text)[0]
    # 标签顺序：LABEL_0 = 左派，LABEL_1 = 中立，LABEL_2 = 右派
    # 给左派分配 -1，右派 +1，中立 0，计算加权平均
    weights = {"LABEL_0": -1.0, "LABEL_1": 0.0, "LABEL_2": 1.0}
    score = sum([entry["score"] * weights[entry["label"]] for entry in result])
    return round(score, 4)

# 示例
text = "We need universal healthcare and stronger gun control laws."
score = get_political_bias_score(text)
print("政治偏见得分:", score)
'''
'''
from transformers import pipeline
import os

# 设置 huggingface 镜像（中国大陆用户建议开启）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 创建文本分类管道，开启返回所有得分
classifier = pipeline("text-classification", model="premsa/political-bias-prediction-allsides-BERT", return_all_scores=True)

def get_political_bias_score(text):
    result = classifier(text)[0]  # 得到一个列表，包含3个类别及其得分
    # 标签含义：LABEL_0 = 左派，LABEL_1 = 中立，LABEL_2 = 右派
    label_map = {"LABEL_0": "左派", "LABEL_1": "中立", "LABEL_2": "右派"}
    weights = {"LABEL_0": -1.0, "LABEL_1": 0.0, "LABEL_2": 1.0}
    score = sum(entry["score"] * weights[entry["label"]] for entry in result)
    print("类别得分分布:", [(label_map[entry["label"]], round(entry["score"], 4)) for entry in result])
    return round(score, 4)

# 测试样本
text1 = "Education is a key to individual success and national progress."
text2 = "The economy is expected to grow by 3% this year."

print("偏见得分 1:", get_political_bias_score(text1))  # 应该偏左，接近 -1
print("偏见得分 2:", get_political_bias_score(text2))  # 也应该偏左，接近 -1
'''