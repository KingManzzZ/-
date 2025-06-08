package com.example.llmtest.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.llmtest.entity.DataInfo;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface DGFromModelMapper extends BaseMapper<DataInfo> {
}
