package com.example.llmtest.pojo.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.example.llmtest.pojo.enums.DataSourceEnum;
import com.example.llmtest.pojo.enums.DimensionEnum;
import com.example.llmtest.pojo.enums.QuestionTypeEnum;
import com.example.llmtest.pojo.enums.TransformationTypeEnum;
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
public class DataInfo implements Serializable {

    @TableId(type = IdType.AUTO)
    private Long dataId;

    //逻辑外键
    private Long modelId;
    //逻辑外键
    private Long metricId;

    private String question;

    private String options;

    private String answer;

    private DimensionEnum dimension;

    @TableField("sub_metric_id")
    private Long subMetricId;

    private QuestionTypeEnum questionType;

    private DataSourceEnum dataSource;

    private TransformationTypeEnum transformationType;

    private String transformationDescription;

    private Long originalDataId;

    @TableField(value = "create_time", fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private Timestamp createTime;

    @TableField(value = "update_time", fill = FieldFill.INSERT_UPDATE)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private Timestamp updateTime;

    @TableLogic
    private Integer isDeleted;

    private Integer isTransformed;

}
