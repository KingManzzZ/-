package com.example.llmtest.pojo.vo;

import com.example.llmtest.pojo.enums.DimensionEnum;
import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.sql.Timestamp;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TestInfoVO {
    private Long testId;
    private String name;
    private String modelName;
    private DimensionEnum dimension;
    //针对性能
    //TODO: 查询时考虑是否为空
    private String metricName;

    private String os;
    private String cpu;
    private String gpu;

    private Long count;
    private String testDescription;
    private Double finalScore;
    private String resultDescription;

    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private Timestamp updateTime;

    //针对按指标给分的
    private Map<String, Double> metricScores;
    //针对按题目给分的
    private Double[] singleScores;
}
