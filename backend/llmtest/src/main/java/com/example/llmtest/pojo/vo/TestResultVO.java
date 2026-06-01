package com.example.llmtest.pojo.vo;

import lombok.Data;

import java.util.Map;

@Data
public class TestResultVO {
    private Long testId;
    private Double finalScore;
    private Double[] singleScore;
    private Map<String, Double> metricScores;
}
