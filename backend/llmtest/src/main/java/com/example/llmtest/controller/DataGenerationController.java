package com.example.llmtest.controller;

import com.example.llmtest.exceptionhandler.R;
import com.example.llmtest.exceptionhandler.ReturnCode;
import com.example.llmtest.pojo.dto.GenerationByHandDTO;
import com.example.llmtest.pojo.dto.GenerationByModelDTO;
import com.example.llmtest.pojo.entity.DataInfo;
import com.example.llmtest.pojo.vo.DataInfoVO;
import com.example.llmtest.service.DataGenerationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@Tag(name = "DataGenerationFromModelController:有关智能生成所有操作")
@RestController
@RequestMapping("/dataInfo/generate")
@Slf4j
public class DataGenerationController {
    private final DataGenerationService dataGenerationService;

    public DataGenerationController(DataGenerationService dataGenerationService) {
        this.dataGenerationService = dataGenerationService;
    }

    @Operation(summary = "模型智能生成")
    @PostMapping("/model")
    public R<List<DataInfoVO>> model(@RequestBody GenerationByModelDTO dto) {
        log.info("dto参数为{}", dto);
        List<DataInfoVO> voList = dataGenerationService.generateByModel(dto);
        return R.success(voList);
    }

    @Operation(summary = "手动录入")
    @PostMapping("/hand")
    public R<String> hand(@RequestBody GenerationByHandDTO dto) {
        log.info("dto参数为{}", dto.toString());
        Boolean returnVal = dataGenerationService.generateByHand(dto);
        if (returnVal) {
            return R.success("成功录入");
        } else{
            return R.error(ReturnCode.RC400.getCode(), ReturnCode.RC400.getMsg());
        }
    }

}
