from Assess.long_text_comprehension import context_understanding
from Assess.long_text_comprehension import extract
from Assess.long_text_comprehension import memory_retention

def main(model, qs_list):
    """
    大模型长文本理解能力评测统一入口。
    子指标：上下文理解、信息提取、记忆能力。
    """
    return evaluate(model, qs_list)

def evaluate(model, qs_list):
    # 分组题目
    groups = {
        'context_understanding': [],
        'information_extraction': [],
        'memory_retention': []
    }

    # 精确 metric 映射表（避免模糊 in 匹配导致误分组）
    metric_mapping = {
        'context_understanding': 'context_understanding',
        'information_extraction': 'information_extraction',
        'memory_retention': 'memory_retention',
    }

    for item in qs_list:
        metric = item.get("min_metric", "").strip()
        target_group = metric_mapping.get(metric)

        if target_group:
            groups[target_group].append(item)
        else:
            # 模糊匹配兜底（保持向后兼容）
            metric_lower = metric.lower()
            if "context" in metric_lower and "extract" not in metric_lower:
                groups['context_understanding'].append(item)
            elif "extract" in metric_lower:
                groups['information_extraction'].append(item)
            elif "memory" in metric_lower or "retention" in metric_lower:
                groups['memory_retention'].append(item)
            else:
                groups['context_understanding'].append(item)

    all_response = []
    scores = {
        'context_understanding': 0.0,
        'information_extraction': 0.0,
        'memory_retention': 0.0
    }

    # 基础权重配置
    base_weights = {
        'context_understanding': 0.4,
        'information_extraction': 0.3,
        'memory_retention': 0.3
    }

    # 子模块映射
    module_mapping = {
        'context_understanding': context_understanding,
        'information_extraction': extract,
        'memory_retention': memory_retention,
    }

    # 执行各子指标评测
    active_keys = []
    for key in groups:
        if not groups[key]:
            print(f"指标 {key} 没有测试数据，将重新分配其权重。")
            continue
        try:
            print(f"开始评测 [{key}] (数量: {len(groups[key])})")
            resp, score = module_mapping[key].evaluate(model, groups[key])
            scores[key] = score
            all_response.extend(resp)
            active_keys.append(key)
        except Exception as e:
            print(f"{key} 评测出错: {e}")

    # 按照 dataId 排序
    all_response.sort(key=lambda x: x.get("dataId", 0))

    # 动态权重归一化：仅在有数据的类别间按比例分配
    if active_keys:
        total_active_weight = sum(base_weights[k] for k in active_keys)
        normalized_weights = {k: base_weights[k] / total_active_weight for k in active_keys}
        final_score = sum(scores[k] * normalized_weights[k] for k in active_keys)
    else:
        final_score = 0.0

    final_report = {
        'context_understanding': round(scores['context_understanding'], 2),
        'information_extraction': round(scores['information_extraction'], 2),
        'memory_retention': round(scores['memory_retention'], 2),
        'final_score': round(final_score, 2)
    }

    print(f"\n长文本理解能力评测完成！最终得分: {final_report['final_score']}")
    return all_response, final_report

if __name__ == "__main__":
    # 模拟测试
    test_qs = [
        {'rowIdx': 0, 'question': '这是一段长文本...', 'answer': '答案', 'min_metric': 'context_understanding'}
    ]
    responses, report = evaluate("DeepSeek-V3", test_qs)
    print(report)

