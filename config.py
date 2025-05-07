def call_api(model:str , prompt:str , retry_limit=3) -> str:
    from openai import OpenAI
    import time
    setting = {
        "DeepSeek-V3": {
            "model_name": "deepseek-ai/DeepSeek-V3",
            "api_key": "sk-rwqwkqbrvrvfyhwqorevlobgotosfuywedkijzwzzqqlpwwh",
            "base_url": "https://api.siliconflow.cn/v1"},
        "qwen-plus": {
            "model_name": "qwen-plus",
            "api_key": "sk-ba1ce076db94414a913606b7887d1539",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"}
    }
    client = OpenAI(api_key=setting[model]["api_key"],
                    base_url=setting[model]["base_url"])
    for attempt in range(retry_limit):
        try:
            response = client.chat.completions.create(
                model=setting[model]["model_name"], 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,  # 控制生成随机性
                top_p=0.95,
                presence_penalty=0.6  # 降低重复短语概率
            )
            return response.choices[0].message.content
        except Exception as e:
            #print(f"API调用失败（第{attempt + 1}次重试）: {e}")
            time.sleep(4)  # 错误等待时间
    return ""  # 重试失败返回空值
