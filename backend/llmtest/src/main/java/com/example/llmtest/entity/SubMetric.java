package com.example.llmtest.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("sub_metric")
public class SubMetric {

    @TableId(type = IdType.AUTO)
    @TableField("sub_metric_id")
    private Long subMetricId;

    @TableField("sub_metric_name")
    private String subMetricName;

    @TableField("sub_metric_description")
    private String subMetricDescription;
}
