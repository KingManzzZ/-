package com.example.llmtest.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.llmtest.pojo.entity.Metric;
import org.apache.ibatis.annotations.Select;

public interface ModelMapper extends BaseMapper<Metric> {

    @Select("SELECT model_id FROM model WHERE model_name = #{modelName}")
    Long selectIdByName(String modelName);

    @Select("SELECT model_name FROM model WHERE model_id = #{modelId}")
    String selectNameById(Long modelId);
}
