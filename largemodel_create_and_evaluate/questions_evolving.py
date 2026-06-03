import json
import re
import sys
import threading
import asyncio
from typing import Dict, List, Tuple
import queue
import logging
from logging.handlers import QueueHandler, QueueListener
import time
from openai import AsyncOpenAI

# 统一导入处理
import import_utils
from largemodel_create_and_evaluate.promot_hub import prompt_hub
from config import api_setting


class ThreadSafeLogger:
    def __init__(self):
        self.log_queue = queue.Queue(-1)
        self.setup_logger()

    def setup_logger(self):
        # 创建根日志记录器
        self.logger = logging.getLogger('QuestionsEvolution')
        self.logger.setLevel(logging.INFO)

        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
        )

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # 创建队列监听器
        self.queue_handler = QueueHandler(self.log_queue)
        self.logger.addHandler(self.queue_handler)

        # 启动队列监听器
        self.listener = QueueListener(self.log_queue, console_handler)
        self.listener.start()

    def get_logger(self):
        return self.logger

class ModelManager:
    def __init__(self):
        self.models = ["DeepSeek-V3", "Qwen-max", 'yi-lightning', "ERNIE-4.0-8k", "gpt-4o-mini"]
        # 使用信号量来控制模型并发访问频率（每个模型同时只允许一个请求，类似原有独占逻辑）
        self.model_semaphores = {model: asyncio.Semaphore(1) for model in self.models}

        self.model_status = {
            model: {
                "status": True,
                "fail_count": 0
            } for model in self.models
        }
        self.lock = threading.Lock()

    def get_active_models(self) -> List[str]:
        """获取当前所有可用的模型名称"""
        with self.lock:
            return [m for m, status in self.model_status.items() if status["status"]]

    def report_status(self, model: str, success: bool, connect_status: bool = True):
        """更新模型状态"""
        with self.lock:
            if success:
                # 成功时重置失败计数
                self.model_status[model]["fail_count"] = 0
                self.model_status[model]["status"] = True
            else:
                # 失败时增加失败计数
                self.model_status[model]["fail_count"] += 1
                if self.model_status[model]["fail_count"] >= 2 or not connect_status:
                    # 禁用模型
                    self.model_status[model]["status"] = False
                    print(f"模型 {model} 已被暂时禁用")

class QuestionsEvolution:
    def __init__(self):
        self.evolution_types = {
            "rewrite": self.paraphrase_evolutions,
            "add_noise": self.addnoise_evolutions,
            "reverse_polarity": self.reversepolar_evolutions,
            "substitute": self.alternative_evolutions,
            "complicate": self.complex_evolutions
        }
        self.questions = []
        self.model_manager = ModelManager()
        self.logger = ThreadSafeLogger().get_logger()
        # 初始化异步客户端
        self.clients = self._init_clients()

    def _init_clients(self) -> Dict[str, AsyncOpenAI]:
        clients = {}
        for model, config in api_setting.items():
            kwargs = {"api_key": config["api_key"]}
            if "base_url" in config:
                kwargs["base_url"] = config["base_url"]
            clients[model] = AsyncOpenAI(**kwargs)
        return clients

    def extract_text(self, pending_part: Dict) :
        ls=[]
        for k,v in pending_part.items():
            if k == "question":
                ls.append(v)
            elif k == "options":
                ls.append(v)
            elif k == "answer":
                ls.append(v)
            elif k == "type_index":
                ls.append(v)
        return ls,pending_part["rowIdx"]

    def build_prompt(self, content: list, evolution_type: str) -> List[Dict[str, str]]:
        q_type = content[-1]
        system_role = prompt_hub['roles']['question_evolver']
        task_desc = prompt_hub['task_descriptions']['evolution']

        user_base = f"{task_desc}\n你需要根据我给出的题目和变形要求进行处理，保持题目答案不变。"

        # 获取更具体的变形要求
        evolution_instruction = self.evolution_types.get(evolution_type)()

        # 获取格式示例
        format_demo = prompt_hub.get('demo', {}).get(q_type, "")

        # 构建待处理题目的描述
        question_desc = ""
        if q_type == "choice":
            question_desc = f"原题目：{content[0]}\n选项：{content[1]}\n正确答案：{content[2]}"
        elif q_type in ("judgment", "short_answer"):
            question_desc = f"原题目：{content[0]}\n正确答案：{content[1]}"
        elif q_type == "compare_question" and isinstance(content[0], dict):
            question_desc = f"原题目 A：{content[0].get('A', '')}\n原题目 B：{content[0].get('B', '')}"
        elif q_type == "only_question":
            question_desc = f"原题目：{content[0]}"

        user_content = f"{user_base}\n\n{question_desc}\n\n变形要求：\n{evolution_instruction}\n\n{format_demo}"

        return [
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_content}
        ]

    @staticmethod
    def _parse_response(response: str):
        """使用 raw_decode 精确提取第一个 JSON 对象"""
        try:
            clean_res = response.strip()
            if clean_res.startswith("```json"):
                clean_res = clean_res.split("```json")[1].split("```")[0].strip()
            elif clean_res.startswith("```"):
                clean_res = clean_res.split("```")[1].split("```")[0].strip()

            start = clean_res.find('{')
            if start == -1:
                return None

            decoder = json.JSONDecoder()
            result, _ = decoder.raw_decode(clean_res, start)

            # 统一返回 dict：如果解析出 list，取第一个 dict 元素
            if isinstance(result, dict):
                return result
            elif isinstance(result, list) and result:
                first_item = result[0]
                return first_item if isinstance(first_item, dict) else None
            return None

        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")
            return None
        except Exception as e:
            print(f"解析响应时发生未预期错误: {e}")
            return None

    async def async_call_api(self, model: str, messages: List[Dict]) -> str:
        """异步调用 API"""
        try:
            client = self.clients[model]
            config = api_setting[model]
            response = await client.chat.completions.create(
                model=config["model_name"],
                messages=messages,
                temperature=0.95,  # 提高随机性以获得更高程度的变形
                presence_penalty=0.6,  # 鼓励模型产生更多样化的表达
                frequency_penalty=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"API 异步调用异常 ({model}): {e}")
            return "ERROR"

    async def evolve_task(self, item: Dict, semaphore: asyncio.Semaphore, model: str):
        """单个演进任务的异步封装"""
        async with semaphore:
            pending_part = item['question0']
            evolve_type = item['evolve_type']
            content, idx = self.extract_text(pending_part)
            messages = self.build_prompt(content, evolve_type)

            self.logger.info(f"使用模型 {model} 处理题目 {idx} (类型: {evolve_type})")

            # 这里可以根据需要增加重试逻辑
            response_text = await self.async_call_api(model, messages)

            if response_text == "ERROR":
                self.model_manager.report_status(model, False, False)
                item['status'] = "fail"
                return item

            question_data = self._parse_response(response_text)
            if question_data is not None:
                self.model_manager.report_status(model, True)
                question_data["rowIdx"] = int(idx)
                question_data['model'] = model
                question_data["status"] = "success"
                return question_data
            else:
                self.model_manager.report_status(model, False)
                item['status'] = "fail"
                return item

    async def process_all_async(self, data: List[Dict], max_retries: int = 3):
        """使用异步队列和任务池处理所有数据，带最大重试限制"""
        results = []
        pending_items = list(data)

        # 为每个 item 初始化重试计数
        for item in pending_items:
            if '_retry_count' not in item:
                item['_retry_count'] = 0

        while pending_items:
            active_models = self.model_manager.get_active_models()
            if not active_models:
                self.logger.error("无可用模型，终止处理")
                # 将剩余项标记为失败
                for item in pending_items:
                    item['status'] = 'fail'
                    results.append(item)
                break

            tasks = []
            for i, item in enumerate(pending_items):
                model = active_models[i % len(active_models)]
                sem = self.model_manager.model_semaphores[model]
                tasks.append(self.evolve_task(item, sem, model))

            batch_results = await asyncio.gather(*tasks)

            # 检查结果，超过重试上限的直接标记失败
            pending_items = []
            for res in batch_results:
                if res and res.get('status') == 'success':
                    # 清理内部字段后加入结果
                    res.pop('_retry_count', None)
                    results.append(res)
                else:
                    retry_count = res.get('_retry_count', 0) + 1 if res else max_retries
                    if retry_count >= max_retries:
                        self.logger.warning(f"题目重试 {max_retries} 次仍失败，跳过")
                        if res:
                            res['status'] = 'fail'
                            res.pop('_retry_count', None)
                            results.append(res)
                    else:
                        if res:
                            res['_retry_count'] = retry_count
                        pending_items.append(res)

            if pending_items:
                self.logger.info(f"等待重试项数量: {len(pending_items)}")
                await asyncio.sleep(1)

        return results

    def final(self, data):
        """入口函数，驱动异步事件循环"""
        try:
            # 运行异步处理逻辑
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            all_results = loop.run_until_complete(self.process_all_async(data))
            loop.close()

            if not all_results and data:
                return "处理失败，无有效结果"

            all_results.sort(key=lambda x: x['rowIdx'])
            return all_results
        except Exception as e:
            self.logger.error(f"Final 过程出错: {e}")
            return str(e)

    # 以下为各进化类型的提示词生成方法
    @staticmethod
    def paraphrase_evolutions() -> str:
        return prompt_hub['generator']['rewrite']

    @staticmethod
    def addnoise_evolutions() -> str:
        return prompt_hub['generator']['add_noise']

    @staticmethod
    def reversepolar_evolutions() -> str:
        return prompt_hub['generator']['reverse_polarity']

    @staticmethod
    def alternative_evolutions() -> str:
        return prompt_hub['generator']['substitute']

    @staticmethod
    def complex_evolutions() -> str:
        return prompt_hub['generator']['complicate']

if __name__ == '__main__':
    #data = json.loads(sys.argv[1])
    data=[{
    "question0": {
        "rowIdx": 0,
        "question": "《红楼梦》中'林黛玉焚稿断痴情'的章节发生在哪个传统节日期间？",
        "options": [
            "A: 端午节",
            "B: 重阳节",
            "C: 元宵节",
            "D: 清明节"
        ],
        "answer": "B",
        "type_index": "choice"
    },
    "evolve_type": "add_noise"
},
{
    "question0":{
        "rowIdx": 1,
        "question": "太阳系中木星的卫星数量超过80颗",
        "answer": "正确",
        "type_index": "judgment"
    },
    "evolve_type":"add_noise"
}
]
    evolve = QuestionsEvolution()
    result = evolve.final(data)
    print(result)