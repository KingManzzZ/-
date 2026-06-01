package com.example.llmtest.pojo.dto;

import lombok.Data;

import java.io.Serializable;

@Data
public class GenerationByHandDTO implements Serializable {
    private String dimension;
    private String metric;
    private String subMetric;
    private String questionType;
    private String question;
    private String options;
    private String answer;
}
