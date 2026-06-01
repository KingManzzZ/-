package com.example.llmtest.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.example.llmtest.exceptionhandler.R;
import com.example.llmtest.pojo.dto.TestDTO;
import com.example.llmtest.pojo.vo.DataInfoVO;
import com.example.llmtest.pojo.vo.TestExactVO;
import com.example.llmtest.pojo.vo.TestResultVO;
import com.example.llmtest.pojo.vo.TestInfoVO;
import com.example.llmtest.service.TestService;
import com.example.llmtest.utils.TestDataGenerator;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Tag(name = "测试")
@RestController
@RequestMapping("/test")
@Slf4j
public class TestController {

    private final TestService testService;
    private final TestDataGenerator testDataGenerator;
    public TestController(TestService testService, TestDataGenerator testDataGenerator) {
        this.testService = testService;
        this.testDataGenerator = testDataGenerator;
    }

    @Operation(summary = "系统响应效率测试和公平性测试")
    @PostMapping("/test1")
    public R<TestResultVO> srTest(@RequestBody TestDTO dto) {
        TestResultVO vo = testService.questionTest(dto);
        return R.success(vo);
    }

    @Operation(summary = "剩余指标测试")
    @PostMapping("/test2")
    public R<TestResultVO> otherTest(@RequestBody TestDTO dto) {
        TestResultVO vo = testService.metricTest(dto);
        return R.success(vo);
    }

    @PostMapping("/autoGenerate")
    public R<List<DataInfoVO>> generate(String dimension, String metric, Integer totalQuestions) {
        List<DataInfoVO> list = testDataGenerator.generateData(dimension, metric, totalQuestions);
        return R.success(list);
    }

    @Operation(summary = "分页查询所有测试信息")
    @GetMapping("/select")
    public R<IPage<TestInfoVO>> selectAll(@RequestParam Long pageNum) {
        IPage<TestInfoVO> vo = testService.selectRoughByPage(pageNum);
        return R.success(vo);
    }

    @Operation(summary = "查询一次测试的详细信息")
    @GetMapping("select/{testId}")
    public R<List<TestExactVO>> selectExact(@PathVariable Long testId) {
        List<TestExactVO> vo = testService.selectExactByList(testId);
        return R.success(vo);
    }
}
