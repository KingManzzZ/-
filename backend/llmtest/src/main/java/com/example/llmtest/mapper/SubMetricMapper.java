package com.example.llmtest.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.llmtest.entity.SubMetric;
import org.apache.ibatis.annotations.Select;

public interface SubMetricMapper extends BaseMapper<SubMetric> {
    @Select("SELECT sub_metric_id FROM sub_metric WHERE sub_metric_name = #{name}")
    Long selectIdByName(String name);
}
