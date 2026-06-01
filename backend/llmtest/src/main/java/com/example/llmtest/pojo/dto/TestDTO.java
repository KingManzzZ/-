package com.example.llmtest.pojo.dto;

import lombok.Data;

import java.util.List;

@Data
public class TestDTO {
    private String testName;
    private String modelName;
    private String dimension;
    private String metric;

    private String os;
    private String cpu;
    private String gpu;

    private List<Long> questionList;
}
