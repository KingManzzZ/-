package com.example.llmtest.pojo.entity;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SubMetric implements Serializable {

    @TableId(type = IdType.AUTO)
    private Long subMetricId;

    private String subMetricName;

    private String subMetricDescription;
}
