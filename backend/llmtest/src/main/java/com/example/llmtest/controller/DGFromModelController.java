package com.example.llmtest.controller;

import com.example.llmtest.service.DGFromModelService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@Tag(name = "DGFromModelController:有关智能生成所有操作")
@RestController
@RequestMapping("/dataInfo/dataGeneration")
public class DGFromModelController {
        private final DGFromModelService dgFromModelService;

        public DGFromModelController(DGFromModelService dgFromModelService) {
            this.dgFromModelService = dgFromModelService;
        }
    @Operation(summary = "智能生成，有五个参数")
    @PostMapping("/submit")
    public String handleUserInput(@RequestBody Map<String, Object> input) {
        return dgFromModelService.processAndSave(input);
    }

}
