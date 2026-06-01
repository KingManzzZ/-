package com.example.llmtest.pojo.dto;

import lombok.Data;

import java.io.Serializable;
import java.util.List;

@Data
public class DGFromModelDTO implements Serializable {
    private String dimension;
    private String metric;
    private String subMetric;
    private List<String> questionType;
    private String example;
    private Long count;
}
