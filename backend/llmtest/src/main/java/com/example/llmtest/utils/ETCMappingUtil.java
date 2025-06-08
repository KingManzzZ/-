package com.example.llmtest.utils;

import lombok.Getter;
import org.springframework.context.annotation.Configuration;

import java.util.HashMap;
import java.util.Map;

@Getter
@Configuration
public class ETCMappingUtil {

    // 维度映射
    private final Map<String, String> dimensionMap = Map.of(
            "性能","performance",
            "可靠性","reliability",
            "安全性","safety",
            "公平性","fairness"
    );

    // 指标映射"
    private final Map<String, String> metricMap = new HashMap<>() {{
        putAll(Map.of(
                "系统响应效率", "system_responsiveness",
                "复杂推理能力", "complex_reasoning_skill",
                "长文本理解能力", "long_text_comprehension_skill"

        ));
        putAll(Map.of(
                "准确性", "accuracy",
                "鲁棒性", "robustness",
                "一致性", "consistency",
                "稳定性", "stability"
        ));
        putAll(Map.of(
                "随机生成样本", "randomly_generated_samples",
                "指令挟持", "command_hijacking",
                "越狱攻击", "jailbreak_attacks",
                "内容扭曲", "content_distortions",
                "提示屏蔽", "prompt_blocking",
                "干扰对话", "disrupt_conversations",
                "黑盒", "black_box",
                "白盒", "white_box"
        ));
        putAll(Map.of(
                "性别", "gender",
                "种族", "race",
                "年龄", "age",
                "宗教", "religion",
                "政治", "politics"
        ));
    }};
    // 子指标映射
    private final Map<String, String> subMetricMap = Map.of(
            "数学推理","mathematical_reasoning",
            "常识逻辑推理","common_sense_logical_reasoning",
            "因果推理","casual_reasoning",
            "信息提取","information_extraction",
            "上下文关联","contextual_relevance",
            "记忆能力","memory_ability"
    );

    // 题型映射
    private final Map<String, Integer> questionTypeMap = Map.of(
            "选择题",0,
            "判断题",1,
            "简答题",2,
            "仅问题",3,
            "问题对比组",4
    );

    // Getter 方法

    /*public Map<String, String> getDimensionMap() {
        return dimensionMap;
    }

    public Map<String, String> getMetricMap() {
        return metricMap;
    }

    public Map<String, String> getSubMetricMap() { return subMetricMap; }

    public Map<String, Integer> getQuestionTypeMap() {
        return questionTypeMap;
    }*/
}

