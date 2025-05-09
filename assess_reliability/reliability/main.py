import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import call_api
import os
import random
import numpy as np
import time


# 配置环境
os.environ['SENTENCE_TRANSFORMERS_HOME'] = 'D:/models/'


# 模型加载
def load_sentence_model():
    """加载句子相似度模型"""
    try:
        model = SentenceTransformer('D:/models/paraphrase-multilingual-MiniLM-L12-v2')
        print("模型从本地加载成功")
        return model
    except:
        print("本地模型不存在，尝试在线下载...")
        try:
            os.environ['HF_MIRROR'] = 'https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models'
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            exit(1)

model = load_sentence_model()

# 数据加载
def load_test_data(file_path):
    """从JSON文件加载测试数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {
                "accuracy": data.get("accuracy_tests", []),
                "consistency": data.get("consistency_tests", []),
                "robustness": data.get("robustness_tests", []),
                "stability": data.get("stability_tests", [])
            }
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return None
    except json.JSONDecodeError:
        print(f"错误：文件 {file_path} 格式不正确")
        return None

# 核心测试功能
class ModelReliabilityTester:
    def __init__(self):
        self.test_results = {
            "accuracy": {
                "total": 0,
                "correct": 0,
                "results": []
            },
            "consistency": {
                "average_similarity": 0,
                "cases": []
            },
            "robustness": {
                "total": 0,
                "passed": 0,
                "results": []
            },
            "stability": {
                "total_requests": 0,
                "successful_requests": 0,
                "response_times": [],
                "requests": []
            }
        }


    def calculate_similarity(self, text1, text2):
        """计算语义相似度"""
        embeddings = model.encode([text1, text2], convert_to_tensor=True)
        return cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]

    # 准确性测试
    def test_accuracy(self, questions):
        print("\n=== 准确性测试 ===")
        for idx, q in enumerate(questions):
            self.test_results["accuracy"]["total"] += 1
            model_answer, response_time, success = call_api(self.model_name, q["question"])
            similarity = self.calculate_similarity(model_answer, q["answer"])
            is_correct = similarity >= 0.7

            if is_correct:
                self.test_results["accuracy"]["correct"] += 1

            result = {
                "index": idx + 1,
                "question": q["question"],
                "model_answer": model_answer,
                "reference_answer": q["answer"],
                "similarity": float(similarity),
                "correct": is_correct
            }

            self.test_results["accuracy"]["results"].append(result)

            print(f"\n问题 {idx + 1}: {q['question']}")
            print(f"模型回答: {model_answer}")
            print(f"参考答案: {q['answer']}")
            print(f"语义相似度: {similarity:.2f}")
            print(f"响应时间: {response_time:.2f}秒")
            print(f"状态: {'正确' if is_correct else '错误'}")

    # 一致性测试
    def test_consistency(self, question_sets):
        print("\n=== 一致性测试 ===")
        for idx, question_set in enumerate(question_sets):
            print(f"\n一致性测试组 {idx + 1}:")
            answers = []
            for q in question_set["variations"]:
                answer, response_time, success = call_api(self.model_name, q)
                answers.append(answer)
                print(f"问题: {q}")
                print(f"模型回答: {answer}")
                print(f"响应时间: {response_time:.2f}秒")

            similarities = []
            for i in range(len(answers)):
                for j in range(i + 1, len(answers)):
                    sim = self.calculate_similarity(answers[i], answers[j])
                    similarities.append(sim)

            avg_sim = np.mean(similarities) if similarities else 0
            result = {
                "questions": question_set["variations"],
                "answers": answers,
                "similarities": [float(s) for s in similarities],
                "average_similarity": float(avg_sim)
            }

            self.test_results["consistency"]["cases"].append(result)

            print(f"平均一致性相似度: {avg_sim:.2f}")

    # 鲁棒性测试
    def add_noise(self, text, noise_level=0.1):
        """添加随机噪声到文本"""
        chars = list(text)
        for _ in range(int(len(chars) * noise_level)):
            idx = random.randint(0, len(chars) - 1)
            chars[idx] = chr(random.randint(97, 122))  # 随机小写字母
        return "".join(chars)

    def test_robustness(self, questions):
        """测试输入噪声下的表现"""
        print("\n=== 鲁棒性测试 ===")
        for idx, q in enumerate(questions):
            self.test_results["robustness"]["total"] += 1
            noisy_q = self.add_noise(q["question"])
            model_answer, response_time, success = call_api(self.model_name, noisy_q)
            similarity = self.calculate_similarity(model_answer, q["answer"])
            passed = similarity >= 0.5

            if passed:
                self.test_results["robustness"]["passed"] += 1

            result = {
                "index": idx + 1,
                "original_question": q["question"],
                "noisy_question": noisy_q,
                "model_answer": model_answer,
                "reference_answer": q["answer"],
                "similarity": float(similarity),
                "passed": passed
            }

            self.test_results["robustness"]["results"].append(result)

            print(f"\n问题 {idx + 1}: {q['question']}")
            print(f"噪声问题: {noisy_q}")
            print(f"模型回答: {model_answer}")
            print(f"参考答案: {q['answer']}")
            print(f"语义相似度: {similarity:.2f}")
            print(f"响应时间: {response_time:.2f}秒")
            print(f"状态: {'通过' if passed else '失败'}")

    # 稳定性测试
    def test_stability(self, questions, num_requests=20, interval=0.5):
        """测试模型在连续请求下的稳定性"""
        print(f"\n=== 稳定性测试 (共{num_requests}次请求) ===")
        for i in range(num_requests):
            # 随机选择一个问题或使用默认问题
            if questions and isinstance(questions, list):
                question = random.choice(questions)["question"]
            else:
                question = f"测试稳定性问题 {i + 1}"

            # 添加随机变化
            if random.random() > 0.5:
                question = self.add_noise(question, noise_level=0.05)

            print(f"\n请求 {i + 1}/{num_requests}: {question}")

            model_answer, response_time, success = call_api(self.model_name, question)

            result = {
                "request_number": i + 1,
                "question": question,
                "model_answer": model_answer,
                "response_time": float(response_time),
                "success": success
            }

            self.test_results["stability"]["requests"].append(result)

            if success:
                print(f"模型回答: {model_answer}")
            else:
                print(f"状态: 失败（{model_answer}）")

            print(f"响应时间: {response_time:.2f}秒")
            time.sleep(interval)

        self.test_results["stability"]["total_requests"] = num_requests
        self.test_results["stability"]["successful_requests"] = sum(1 for r in self.test_results["stability"]["requests"] if r["success"])
        self.test_results["stability"]["response_times"] = [r["response_time"] for r in self.test_results["stability"]["requests"]]

        response_times = self.test_results["stability"]["response_times"]
        half = len(response_times) // 2
        first_half_avg = np.mean(response_times[:half]) if half > 0 else 0
        second_half_avg = np.mean(response_times[half:]) if half > 0 else 0
        performance_degradation = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0

        self.test_results["stability"]["performance_degradation"] = float(performance_degradation)

    # 生成报告
    def generate_report(self):
        print("\n=== 逐题测试结果报告 ===")

        # 输出准确性测试的每道题
        print("\n--- 准确性测试详细结果 ---")
        for r in self.test_results["accuracy"]["results"]:
            print(f"[准确性] 问题: {r['question']}")
            print(f"模型回答: {r['model_answer']}")
            print(f"参考答案: {r['reference_answer']}")
            print(f"相似度: {r['similarity']:.2f}")
            print(f"结果: {'正确' if r['correct'] else '错误'}")

        # 输出一致性测试的每道题
        print("\n--- 一致性测试详细结果 ---")
        for i, case in enumerate(self.test_results["consistency"]["cases"]):
            print(f"[一致性] 测试组 {i+1}:")
            for j, q in enumerate(case["questions"]):
                print(f"问题 {j+1}: {q}")
                print(f"模型回答: {case['answers'][j]}")
            print(f"相似度列表: {[f'{s:.2f}' for s in case['similarities']]}")
            print(f"平均相似度: {case['average_similarity']:.2f}")

        # 输出鲁棒性测试的每道题
        print("\n--- 鲁棒性测试详细结果 ---")
        for r in self.test_results["robustness"]["results"]:
            print(f"[鲁棒性] 原始问题: {r['original_question']}")
            print(f"噪声问题: {r['noisy_question']}")
            print(f"模型回答: {r['model_answer']}")
            print(f"参考答案: {r['reference_answer']}")
            print(f"相似度: {r['similarity']:.2f}")
            print(f"结果: {'通过' if r['passed'] else '失败'}")

        # 输出稳定性测试的每道题
        print("\n--- 稳定性测试详细结果 ---")
        for r in self.test_results["stability"]["requests"]:
            print(f"[稳定性] 请求 {r['request_number']}:")
            print(f"问题: {r['question']}")
            print(f"模型回答: {r['model_answer']}")
            print(f"响应时间: {r['response_time']:.2f}秒")
            print(f"状态: {'成功' if r['success'] else '失败'}")

        # 汇总报告
        print("\n=== 汇总可靠性报告 ===")
        acc = self.test_results["accuracy"]
        rob = self.test_results["robustness"]
        stab = self.test_results["stability"]

        if acc["total"] > 0:
            acc_rate = acc["correct"] / acc["total"]
            print(f"\n准确性: {acc['correct']}/{acc['total']} ({acc_rate:.1%})")

        if rob["total"] > 0:
            rob_rate = rob["passed"] / rob["total"]
            print(f"\n鲁棒性: {rob['passed']}/{rob['total']} ({rob_rate:.1%})")

        if stab["total_requests"] > 0:
            success_rate = stab["successful_requests"] / stab["total_requests"]
            avg_time = np.mean(stab["response_times"]) if stab["response_times"] else 0
            print(f"\n稳定性:")
            print(f"请求总数: {stab['total_requests']}")
            print(f"成功率: {success_rate:.1%}")
            print(f"平均响应时间: {avg_time:.2f}秒")

        # 综合评分
        if (acc["total"] > 0 and rob["total"] > 0 and stab["total_requests"] > 0):
            acc_rate = acc["correct"] / acc["total"]
            avg_sim = np.mean([c["average_similarity"] for c in self.test_results["consistency"]["cases"] if c["average_similarity"] > 0])
            rob_rate = rob["passed"] / rob["total"]
            success_rate = stab["successful_requests"] / stab["total_requests"]
            perf_deg = stab["performance_degradation"]

            stability_score = success_rate * 0.7 + (1 - min(perf_deg, 1)) * 0.3
            overall_score = (acc_rate * 0.4 + avg_sim * 0.2 + rob_rate * 0.2 + stability_score * 0.2) * 100
            print(f"\n综合可靠性评分: {overall_score:.1f}/100")
        else:
            print("\n警告: 部分测试数据不足，无法计算综合评分")

        self.save_results_to_json()

    def save_results_to_json(self, file_path="test_results.json"):
        """将测试结果保存到JSON文件"""
        try:
            results_to_save = {
                "accuracy": {
                    "total": self.test_results["accuracy"]["total"],
                    "correct": self.test_results["accuracy"]["correct"],
                    "accuracy_rate": self.test_results["accuracy"]["correct"] / self.test_results["accuracy"]["total"]
                    if self.test_results["accuracy"]["total"] > 0 else 0,
                    "results": self.test_results["accuracy"]["results"]
                },
                "consistency": {
                    "average_similarity": np.mean([c["average_similarity"] for c in self.test_results["consistency"]["cases"]
                                                 if c["average_similarity"] > 0]),
                    "cases": self.test_results["consistency"]["cases"]
                },
                "robustness": {
                    "total": self.test_results["robustness"]["total"],
                    "passed": self.test_results["robustness"]["passed"],
                    "pass_rate": self.test_results["robustness"]["passed"] / self.test_results["robustness"]["total"]
                    if self.test_results["robustness"]["total"] > 0 else 0,
                    "results": self.test_results["robustness"]["results"]
                },
                "stability": {
                    "total_requests": self.test_results["stability"]["total_requests"],
                    "successful_requests": self.test_results["stability"]["successful_requests"],
                    "average_response_time": np.mean(self.test_results["stability"]["response_times"])
                    if self.test_results["stability"]["response_times"] else 0,
                    "performance_degradation": self.test_results["stability"]["performance_degradation"],
                    "requests": self.test_results["stability"]["requests"]
                }
            }

            # 综合评分
            if (self.test_results["accuracy"]["total"] > 0 and
                    self.test_results["robustness"]["total"] > 0 and
                    self.test_results["stability"]["total_requests"] > 0):
                acc_rate = results_to_save["accuracy"]["accuracy_rate"]
                avg_sim = results_to_save["consistency"]["average_similarity"]
                rob_rate = results_to_save["robustness"]["pass_rate"]
                stab_score = (results_to_save["stability"]["successful_requests"] / results_to_save["stability"]["total_requests"]) * 0.7 + \
                            (1 - min(results_to_save["stability"]["performance_degradation"], 1)) * 0.3
                results_to_save["overall_score"] = (acc_rate * 0.4 + avg_sim * 0.2 + rob_rate * 0.2 + stab_score * 0.2) * 100

            # 写入JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results_to_save, f, ensure_ascii=False, indent=4)
            print(f"\n测试结果已保存到 {file_path}")
        except Exception as e:
            print(f"\n保存测试结果时出错: {e}")


def main():
    # 1. 加载测试数据
    test_data = load_test_data("test_data.json")
    if not test_data:
        print("无法加载测试数据，程序终止")
        return

    # 2. 初始化测试器
    tester = ModelReliabilityTester()

    # 3. 执行测试
    tester.test_accuracy(test_data["accuracy"])
    tester.test_consistency(test_data["consistency"])
    tester.test_robustness(test_data["robustness"])
    tester.test_stability(test_data.get("robustness", []), num_requests=20)

    # 4. 生成报告
    tester.generate_report()

if __name__ == "__main__":
    main()
