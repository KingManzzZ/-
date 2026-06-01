package com.example.llmtest.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Select;
import com.example.llmtest.pojo.entity.Metric;

public interface MetricMapper extends BaseMapper<Metric> {

    @Select("SELECT metric_id FROM metric WHERE metric_name = #{name}")
    Long selectIdByName(String name);

    @Select("SELECT metric_name FROM metric WHERE metric_id = #{id}")
    String selectNameById(Long id);
}

