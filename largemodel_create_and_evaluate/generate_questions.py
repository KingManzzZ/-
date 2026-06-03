import random
import json
import sys
import os
from typing import Dict, List

# 统一导入处理
import import_utils
from config import call_api
from largemodel_create_and_evaluate.requirements_of_safety import requirements_of_safety
from largemodel_create_and_evaluate.promot_hub import prompt_hub


def show_progress(current: int, total: int):
    """本地进度条显示"""
    if total == 0:
        return

    percentage = round((current / total) * 100, 1)
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)

    sys.stdout.write(f'\r进度 [{bar}] {percentage}% ({current}/{total})')
    sys.stdout.flush()

    if current == total:
        print()  # 完成时换行

class QuestionBankGenerator:
    def __init__(self):
        #总储存
        self.generated_questions = []

    def build_prompt(self, type: str, domain: str, example: str) -> List[Dict[str, str]]:
        content_demo = f"参考示例内容：{example}\n" if example else ""
        system_role = prompt_hub['roles']['question_generator']
        task_desc = prompt_hub['task_descriptions']['generation']
        
        # 从 prompt_hub 中获取对应的题型要求
        type_instruction = prompt_hub['generation_types'].get(type, "")

        user_content = f"{task_desc}\n待生成题目的领域: {domain}\n{content_demo}{type_instruction}"

        keyword_list = ["安全", "safe", "safety", '安全性']
        safe = any(keyword in domain for keyword in keyword_list)

        if safe:
            special_prompt = random.choice(requirements_of_safety)
            user_content += f"\n生成的结果应适用测试以下场景中的其中一种:\n{special_prompt}"

        return [
            {"role": "system", "content": system_role},
            {"role": "user", "content": user_content}
        ]

    def generate_question(self, domain: str, weights_set: list[int], count: int, example: str, connect_model: str) -> bool | None:
        # 权重,格式为[0,0,0,0,0]依次为选择，判断，简答，仅题目，仅题目组
        if weights_set is not None:
            types_list = ['choice', 'judgment', 'short_answer', 'only_question', 'compare_question']
            q_type = random.choices(types_list, weights=weights_set)[0]

            # 构建消息列表并调用API
            messages = self.build_prompt(q_type, domain, example)

            # 显示 API 调用信息（仅显示一次 System Prompt）
            if count == 0:
                print("\n" + "="*80)
                print("📤 API 调用信息")
                print("="*80)
                print(f"\n【System 角色】")
                print("-"*80)
                print(messages[0]['content'][:500] + "..." if len(messages[0]['content']) > 500 else messages[0]['content'])
                print(f"\n【User 角色】")
                print("-"*80)
                print(messages[1]['content'][:500] + "..." if len(messages[1]['content']) > 500 else messages[1]['content'])
                print("\n" + "="*80 + "\n")

            response = call_api(connect_model, messages=messages)

            # 显示模型原始响应
            print(f"\n📥 第 {count+1} 个题目 - 模型原始响应")
            print("-"*80)
            print(response[:300] + "..." if len(response) > 300 else response)
            print("-"*80)

        else:
            return False

        # 处理有效响应
        if response:
            question = self._parse_response(response)
            pre_question = self._validate_question(q_type, question)
            if pre_question:
                key_order = list(pre_question.keys())
                pre_question["rowIdx"] = count
                key_order.insert(0, "rowIdx")
                new_question = {key: pre_question[key] for key in key_order}
                self.generated_questions.append(new_question)
                return True
        return False

    @staticmethod
    def _parse_response(response: str) -> Dict:
        """使用 json.JSONDecoder.raw_decode 精确提取第一个 JSON 对象"""
        try:
            if not response:
                return {}

            clean_res = response.strip()

            # 处理 Markdown 代码块包裹
            if clean_res.startswith("```"):
                if "json" in clean_res[:10]:
                    clean_res = clean_res.split("json", 1)[1]
                else:
                    clean_res = clean_res.split("```", 1)[1]
                clean_res = clean_res.rsplit("```", 1)[0].strip()

            # 定位第一个 { 的位置
            start = clean_res.find('{')
            if start == -1:
                return {}

            # 使用 raw_decode 精确解析第一个完整 JSON 对象
            decoder = json.JSONDecoder()
            result, _ = decoder.raw_decode(clean_res, start)

            if isinstance(result, dict):
                return result
            return {}

        except json.JSONDecodeError as e:
            print(f"\n❌ JSON 解析失败: {e}")
            print(f"响应内容（前300字）：{response[:300]}")
            return {}
        except Exception as e:
            print(f"\n❌ 解析响应失败: {e}")
            print(f"响应内容（前300字）：{response[:300]}")
            return {}

    def _validate_question(self, q_type: str, question: Dict):
        """验证题目内容的完整性"""
        if not question:
            return None

        required_fields = {
            'choice': ['question', 'options', 'answer'],
            'judgment': ['question', 'answer'],
            'short_answer': ['question', 'answer'],
            'only_question': ['question'],
            'compare_question': ['question']
        }

        fields = required_fields.get(q_type, [])
        if all(field in question for field in fields):
            # 特殊校验逻辑
            if q_type == 'choice':
                options = question.get('options', [])
                if not isinstance(options, list) or len(options) != 4:
                    return None

            question["type_index"] = q_type
            return question
        return None

    def generate_question_bank(self, domain: str, weights_set, example: str, target_count: int, connect_model: str) -> None:
        """生成题目集合，带最大尝试次数防止死循环"""
        generated = 0
        rowidx = 0
        total_attempts = 0
        max_attempts = target_count * 3  # 最多尝试 3 倍目标数量
        consecutive_failures = 0
        max_consecutive_failures = 10

        print(f"\n📝 开始生成题库")
        print(f"   评估维度: {domain}")
        print(f"   目标数量: {target_count} 道\n")

        while generated < target_count:
            if total_attempts >= max_attempts:
                print(f"\n⚠️ 已达最大尝试次数 ({max_attempts})，停止生成。已生成 {generated}/{target_count} 道。")
                break
            if consecutive_failures >= max_consecutive_failures:
                print(f"\n⚠️ 连续失败 {consecutive_failures} 次，停止生成。已生成 {generated}/{target_count} 道。")
                break

            batch_size = min(10, target_count - generated)
            success_count = 0

            for _ in range(batch_size):
                total_attempts += 1
                if self.generate_question(domain, weights_set, rowidx, example, connect_model):
                    success_count += 1
                    consecutive_failures = 0
                    show_progress(generated + success_count, target_count)
                    rowidx += 1
                else:
                    consecutive_failures += 1

            generated += success_count

    def save_to_file(self, filename: str) -> str:
        """保存题库到项目根目录下的 LargeModel_Generate 目录"""
        # 获取项目根目录 (假设该脚本在子目录下)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target_dir = os.path.join(base_dir, "LargeModel_Generate")

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print(f"创建目录: {target_dir}")

        if not filename.endswith(".json"):
            filename += ".json"

        file_path = os.path.join(target_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.generated_questions, f, ensure_ascii=False, indent=2)
        return file_path

    def final(self, data):
        dimension = data['dimension']
        domain = f"有关{dimension}"
        metric = data.get('metric', '')
        sub_metric = data.get('sub_metric', '')
        if metric:
            domain += f"中的{metric}"
        if sub_metric:
            domain += f"的{sub_metric}"

        weights_set = data['weights_set']
        example = data.get('example', '')
        model0 = data.get('model', 'DeepSeek-V3')

        self.generate_question_bank(domain, weights_set, example, data['count'], model0)
        return self.generated_questions


if __name__ == "__main__":
    # 删除命令行参数解析，改为交互式菜单
    generator = QuestionBankGenerator()

    # 显示欢迎信息
    print("\n" + "="*60)
    print("        LLM 题库生成工具")
    print("="*60)

    while True:
        print("\n📋 请选择操作:")
        print("  1. 快速生成 (使用默认配置)")
        print("  2. 自定义生成 (配置各项参数)")
        print("  3. 退出程序")

        choice = input("\n请输入选项 (1/2/3): ").strip()

        if choice == "1":
            # 快速生成模式
            dimension = input("请输入评估维度 (如: 公平性, 安全性): ").strip()
            if not dimension:
                print("❌ 维度不能为空")
                continue

            # 使用默认配置
            data = {
                "dimension": dimension,
                "metric": "",
                "sub_metric": "",
                "count": 5,
                "weights_set": [20, 20, 20, 20, 20],
                "example": "",
                "model": "DeepSeek-V3"
            }

            print(f"\n✓ 开始生成...")
            result = generator.final(data)
            print(f"✓ 生成成功: {len(result)} 道题目")

            # 保存文件
            import time
            output_name = f"{dimension}_{int(time.time())}"
            save_path = generator.save_to_file(output_name)
            print(f"✓ 已保存至: {save_path}\n")
            generator = QuestionBankGenerator()  # 重置生成器

        elif choice == "2":
            # 自定义生成模式
            print("\n🔧 自定义配置")
            print("-" * 60)

            dimension = input("评估维度 (必需): ").strip()
            if not dimension:
                print("❌ 维度不能为空")
                continue

            metric = input("具体指标 (可选，留空跳过): ").strip()
            sub_metric = input("子指标 (可选，留空跳过): ").strip()

            count_str = input("生成数量 (默认: 5): ").strip()
            count = int(count_str) if count_str.isdigit() else 5

            print("\n题型权重配置 (格式: 选择,判断,简答,仅题目,对比)")
            print("示例: 50,50,0,0,0 (仅生成选择和判断题)")
            weights_str = input("权重 (默认: 20,20,20,20,20): ").strip()

            if weights_str:
                try:
                    weights = list(map(int, weights_str.split(',')))
                    if len(weights) != 5:
                        print("❌ 权重必须包含5个数值")
                        continue
                except ValueError:
                    print("❌ 权重格式错误，使用默认值")
                    weights = [20, 20, 20, 20, 20]
            else:
                weights = [20, 20, 20, 20, 20]

            example = input("参考示例 (可选，留空跳过): ").strip()

            print("\n支持的模型:")
            print("  1. DeepSeek-V3 (默认)")
            print("  2. Qwen-max")
            print("  3. yi-lightning")
            print("  4. ERNIE-4.0-8k")
            print("  5. gpt-4o-mini")
            model_choice = input("选择模型 (默认: 1): ").strip()

            models = {
                "1": "DeepSeek-V3",
                "2": "Qwen-max",
                "3": "yi-lightning",
                "4": "ERNIE-4.0-8k",
                "5": "gpt-4o-mini"
            }
            model = models.get(model_choice, "DeepSeek-V3")

            output_name = input("输出文件名 (可选，留空自动生成): ").strip()

            # 执行生成
            data = {
                "dimension": dimension,
                "metric": metric,
                "sub_metric": sub_metric,
                "count": count,
                "weights_set": weights,
                "example": example,
                "model": model
            }

            print(f"\n✓ 开始生成...")
            print(f"  维度: {dimension}" + (f" > {metric}" if metric else ""))
            print(f"  数量: {count} 道题目")
            print(f"  模型: {model}\n")

            result = generator.final(data)
            print(f"\n✓ 生成成功: {len(result)} 道题目")

            # 保存文件
            if output_name:
                save_path = generator.save_to_file(output_name)
            else:
                import time
                output_name = f"{dimension}_{int(time.time())}"
                save_path = generator.save_to_file(output_name)

            print(f"✓ 已保存至: {save_path}\n")
            generator = QuestionBankGenerator()  # 重置生成器

        elif choice == "3":
            print("\n👋 感谢使用，再见！\n")
            break
        else:
            print("❌ 无效选项，请输入 1/2/3")


