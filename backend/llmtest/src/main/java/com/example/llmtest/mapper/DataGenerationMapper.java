package com.example.llmtest.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.llmtest.pojo.entity.DataInfo;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface DataGenerationMapper extends BaseMapper<DataInfo> {
}
