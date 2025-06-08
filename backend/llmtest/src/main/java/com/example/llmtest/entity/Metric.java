package com.example.llmtest.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("metric")
public class Metric {

    @TableId(type = IdType.AUTO)
    @TableField("metric_id")
    private Long metricId;

    @TableField("metric_name")
    private String metricName;

    @TableField("metric_description")
    private String metricDescription;
}
