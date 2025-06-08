package com.example.llmtest.entity;

import com.baomidou.mybatisplus.annotation.*;
import com.example.llmtest.entity.enums.DataSourceEnum;
import com.example.llmtest.entity.enums.DimensionEnum;
import com.example.llmtest.entity.enums.QuestionTypeEnum;
import com.example.llmtest.entity.enums.TransformationTypeEnum;
import com.fasterxml.jackson.annotation.JsonFormat;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.sql.Timestamp;

@Data
@TableName("data_info")
public class DataInfo {

    @Schema(description = "题目id", example = "1")
    @TableId(type = IdType.AUTO)
    @TableField("data_id")
    private Long dataId;

    @TableField("model_id")//foreign key
    private Long modelId;

    private String question;

    @Schema(description = "只有选择题才会有该字段，其他情况为null，不同选项之间用|分隔",
            example = "A.我是A | B.我是B")
    private String options;

    private String answer;

    @Schema(description = "测评维度")
    private DimensionEnum dimension;


    @TableField("metric_id")
    private Long metricId;

    @Schema(description = "专门针对复杂推理能力和长文本理解能力下的子指标设立的")
    @TableField("sub_metric_id")
    private Long subMetricId;

    private QuestionTypeEnum questionType;

    private DataSourceEnum dataSource;

    private TransformationTypeEnum transformationType;

    private String transformationDescription;

    @TableField(value = "entry_date", fill = FieldFill.INSERT)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private Timestamp createTime;

    @TableField(value = "last_updated", fill = FieldFill.INSERT_UPDATE)
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private Timestamp updateTime;

    @TableLogic
    @TableField(value = "is_deleted")
    private Integer isDeleted;

    @TableField(value = "is_transformed")
    private Integer isTransformed;
}
