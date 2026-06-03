import re
from . import causality
from . import logic
from . import math

def main(model, qs_list):
    """
    Standardized entry point for complex reasoning assessment.
    """
    return evaluate(model, qs_list)

def evaluate(model, qs_list):
    """
    统一调度复杂推理评测逻辑
    按 3:3:4 的权重计算最终得分
    """
    # 类别映射和数据分组
    category_map = {
        'casual_reasoning': 'causality',
        'causality': 'causality',
        'common_sense_logical_reasoning': 'logic',
        'mathematical_reasoning': 'math'
    }

    data_groups = {'causality': [], 'logic': [], 'math': []}
    for item in qs_list:
        metric = item.get("min_metric", "")
        group_key = category_map.get(metric)
        if group_key:
            data_groups[group_key].append(item)

    all_response = []
    scores = {'causality': 0.0, 'logic': 0.0, 'math': 0.0}

    # 动态调度的模块映射
    module_mapping = {
        'causality': causality,
        'logic': logic,
        'math': math
    }

    # 基础权重配置
    base_weights = {'causality': 0.3, 'logic': 0.3, 'math': 0.4}

    # 执行子项评估
    active_keys = []
    for key, items in data_groups.items():
        if not items:
            print(f"指标 {key} 没有测试数据，将重新分配其权重。")
            continue

        try:
            print(f"正在执行子指标评测: {key} (数量: {len(items)})")
            run_module = module_mapping[key]
            resp, score = run_module.evaluate(model, items)

            scores[key] = float(score)
            all_response.extend(resp)
            active_keys.append(key)
        except Exception as e:
            print(f"评估子项 {key} 时发生错误: {e}")

    # 按照 dataId 统一排序
    all_response.sort(key=lambda x: x.get("dataId", 0))

    # 动态权重归一化：仅在有数据的类别间按比例分配权重
    if active_keys:
        total_active_weight = sum(base_weights[k] for k in active_keys)
        normalized_weights = {k: base_weights[k] / total_active_weight for k in active_keys}
        final_score = sum(scores[k] * normalized_weights[k] for k in active_keys)
    else:
        final_score = 0.0

    # 汇总最终报告
    final_report = {
        'casual_reasoning': scores['causality'],
        'common_sense_logical_reasoning': scores['logic'],
        'mathematical_reasoning': scores['math'],
        'final_score': round(final_score, 4)
    }

    print(all_response)
    return all_response, final_report

if __name__ == "__main__":
    # 测试代码
    # 示例测试题目列表
    test_qs = [{'rowIdx': 0, 'question': '测试问题', 'options': ['A', 'B'], 'answer': 'A', 'min_metric': 'casual_reasoning'}]
    all_response, result = evaluate("gpt-4o-mini", test_qs)
    print(f"所有响应: {all_response}")
    print(f"综合推理能力得分: {result}")