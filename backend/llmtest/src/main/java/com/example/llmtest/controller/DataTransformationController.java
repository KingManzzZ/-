package com.example.llmtest.controller;

import com.example.llmtest.exceptionhandler.R;
import com.example.llmtest.pojo.dto.TransformationDTO;
import com.example.llmtest.pojo.entity.DataInfo;
import com.example.llmtest.pojo.vo.DataInfoVO;
import com.example.llmtest.service.DataTransformationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.List;


@Tag(name = "DataTransformationController:题目变形")
@RestController
@RequestMapping("/dataInfo/transform")
@Slf4j
public class DataTransformationController {
    private final DataTransformationService dataTransformationService;

    public DataTransformationController(DataTransformationService dataTransformationService) {
        this.dataTransformationService = dataTransformationService;
    }

    @Operation(summary = "题目变形")
    @PostMapping
    public R<List<DataInfoVO>> transform(@RequestBody TransformationDTO dto) {
        List<DataInfoVO> dataInfos = dataTransformationService.transformByModel(dto);
        return R.success(dataInfos);
    }
}
