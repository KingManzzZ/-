package com.example.llmtest.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Select;
import com.example.llmtest.entity.Metric;

public interface MetricMapper extends BaseMapper<Metric> {

    @Select("SELECT metric_id FROM metric WHERE metric_name = #{name}")
    Long selectIdByName(String name);
}

