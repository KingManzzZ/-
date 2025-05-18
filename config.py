def call_api(model,prompt,retry_limit=3) -> str:#输入（setting中的模型，提示内容）
    from setting import api_setting
    from openai import OpenAI
    import time
    for attempt in range(retry_limit):
        try:
            client_args = {"api_key": api_setting[model]["api_key"]}
            if "base_url" in api_setting[model]:
                client_args["base_url"] = api_setting[model]["base_url"]
            client = OpenAI(**client_args)
            response = client.chat.completions.create(
                model=api_setting[model]["model_name"],  
                messages=[{"role": "user", "content": prompt}],
                temperature=0.85,  # 控制生成随机性
                top_p=0.95,
                presence_penalty=0.5  # 降低重复短语概率
            )
            return response.choices[0].message.content
        except Exception as e:
            #print(f"API调用失败（第{attempt + 1}次重试）: {e}")
            time.sleep(4)  # 错误等待时间
    raise ConnectionError("API调用失败")
