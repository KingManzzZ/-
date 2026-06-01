package com.example.llmtest.pojo.dto;

import lombok.Data;

import java.io.Serializable;

@Data
public class DataInfoPageQueryDTO implements Serializable {
    private Long pageNum;
    private String questionType;
    private String dimension;
    private String metric;
    private String subMetric;
}
