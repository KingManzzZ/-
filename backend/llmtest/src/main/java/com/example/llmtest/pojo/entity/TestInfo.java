package com.example.llmtest.pojo.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.example.llmtest.pojo.enums.DimensionEnum;
import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;
import java.sql.Timestamp;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TestInfo implements Serializable {
    @TableId(type = IdType.AUTO)
    private Long testId;
    private String name;
    private Long modelId;
    private DimensionEnum dimension;
    private Long metricId;

    private String os;
    private String cpu;
    private String gpu;

    private Long count;
    private String testDescription;

    private String status;
    private Double finalScore;
    private String resultDescription;

    @TableLogic
    private Integer isDeleted;

    @TableField(value = "create_time", fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private Timestamp createTime;

    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private Timestamp updateTime;
}
