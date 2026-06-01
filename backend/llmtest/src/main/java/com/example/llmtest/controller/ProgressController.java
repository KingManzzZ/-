package com.example.llmtest.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/progress")
public class ProgressController {
    @PostMapping("/update")
    public void receiveProgress(@RequestBody Map<String, Double> requestData) {
        System.out.println(requestData.get("progress"));
    }
}
