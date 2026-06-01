package com.example.llmtest.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.llmtest.pojo.entity.SubMetric;
import org.apache.ibatis.annotations.Select;

public interface SubMetricMapper extends BaseMapper<SubMetric> {
    @Select("SELECT sub_metric_id FROM sub_metric WHERE sub_metric_name = #{name}")
    Long selectIdByName(String name);

    @Select("SELECT sub_metric_name FROM sub_metric WHERE sub_metric_id = #{id}")
    String selectNameById(Long id);
}
