package com.example.llmtest.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.llmtest.pojo.entity.TestInfo;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface TestInfoMapper extends BaseMapper<TestInfo> {
}
