from transformers import pipeline
# 加载情感分析模型
sentiment_model = pipeline("sentiment-analysis")

def normalize_sentiment_score(text):
    # 使用情感分析模型进行预测
    result = sentiment_model(text)
    # 获取情感标签和分数
    sentiment = result[0]['label']  # 'POSITIVE' 或 'NEGATIVE'
    score = result[0]['score']  # 得分，表示置信度
    # 将得分映射到 0-1 之间，负面情感映射为 0，正面情感映射为 1
    if sentiment == "POSITIVE":
        # 正面情感直接返回得分（得分越大表示越积极）
        normalized_score = score
    else:
        # 负面情感反转得分（得分越小表示情感越负面）
        normalized_score = 1 - score
    return normalized_score

'''
# 示例文本
text_1 = "Lowering taxes helps businesses grow and creates jobs."

# 获取情感分数
score_1 = normalize_sentiment_score(text_1)

print(f"Text 1 Normalized Sentiment Score: {score_1}")
'''