package com.example.llmtest.pojo.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import com.example.llmtest.pojo.enums.DimensionEnum;
import lombok.Data;

@Data
public class TestScores {

    @TableId(type = IdType.AUTO)
    private Long id;
    private Long testId;

    private DimensionEnum dimension;
    private String itemType;//question or metric
    private Long itemId;//dataId或metricId(sub)

    private Double score;
}
