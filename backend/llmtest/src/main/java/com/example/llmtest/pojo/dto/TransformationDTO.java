package com.example.llmtest.pojo.dto;

import lombok.Data;

import java.util.List;

@Data
public class TransformationDTO {
    private List<Long> dataIds;
    private String transformationType;
}
