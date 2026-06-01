package com.example.llmtest.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.IService;
import com.example.llmtest.pojo.dto.TestDTO;
import com.example.llmtest.pojo.entity.TestInfo;
import com.example.llmtest.pojo.vo.TestExactVO;
import com.example.llmtest.pojo.vo.TestResultVO;
import com.example.llmtest.pojo.vo.TestInfoVO;

import java.util.List;

public interface TestService extends IService<TestInfo> {

    /**
     * 系统响应效率和公平性测试
     * @return vo
     */
    TestResultVO questionTest(TestDTO dto);

    /**
     * 剩余指标测试
     * @return vo
     */
    TestResultVO metricTest(TestDTO dto);

    /**
     * 分页查询所有测试信息
     * @param pageNum
     * @return
     */
    IPage<TestInfoVO> selectRoughByPage(Long pageNum);

    /**
     * 详细查询测试信息
     * @param testId
     * @return
     */
    List<TestExactVO> selectExactByList(Long testId);
}
