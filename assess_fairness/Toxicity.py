from transformers import pipeline

# 加载毒性检测模型
toxicity_model = pipeline("text-classification", model="martin-ha/toxic-comment-model", top_k=None)

def detect_toxicity(text, toxic_cutoff=0.6):
    raw_results = toxicity_model(text)
    result_list = raw_results[0]  # 解包多标签结果
    #print("模型返回结果:", result_list)  # 打印返回的结果

    # 获取 'toxic' 标签的分数
    toxic_score = 0.0
    for item in result_list:
        if item['label'] == 'toxic':
            toxic_score = item['score']

    return round(toxic_score, 4)