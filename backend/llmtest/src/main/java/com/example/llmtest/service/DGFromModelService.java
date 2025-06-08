package com.example.llmtest.service;

import com.example.llmtest.utils.ETCMappingUtil;
import com.example.llmtest.entity.DataInfo;
import com.example.llmtest.entity.enums.DataSourceEnum;
import com.example.llmtest.entity.enums.DimensionEnum;
import com.example.llmtest.entity.enums.QuestionTypeEnum;
import com.example.llmtest.mapper.DataInfoMapper;
import com.example.llmtest.mapper.MetricMapper;
import com.example.llmtest.mapper.SubMetricMapper;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
public class DGFromModelService {

    private static final String FLASK_URL = "https://test-1-www.u659522.nyat.app:40794/generate";
    private final RestTemplate restTemplate = new RestTemplate();
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final DataInfoMapper dataInfoMapper;
    private final MetricMapper  metricMapper;
    private final SubMetricMapper subMetricMapper;
    private final ETCMappingUtil mappingConfig;

    public DGFromModelService(ETCMappingUtil mapping, DataInfoMapper dataInfoMapper,
                              MetricMapper metricMapper, SubMetricMapper subMetricMapper) {
        this.mappingConfig = mapping;
        this.dataInfoMapper = dataInfoMapper;
        this.metricMapper = metricMapper;
        this.subMetricMapper = subMetricMapper;
    }

    public String processAndSave(Map<String, Object> input) {

        // 构建请求体
        Map<String, Object> requestBody = new HashMap<>();
        String dimensionZh = (String) input.get("dimension");
        String metricZh = (String) input.get("metric");
        String subMetricZh = (String) input.get("sub_metric");
        requestBody.put("dimension", dimensionZh);
        if (subMetricZh == null || subMetricZh.isEmpty()) {
            requestBody.put("metric", metricZh);
        } else {
            requestBody.put("metric", metricZh);
            requestBody.put("sub_metric", subMetricZh);
        }
        @SuppressWarnings("unchecked")
        List<String> questionTypes = (List<String>) input.get("questionType");
        int[] weight = new int[5];
        if (questionTypes != null) {
            for (String type : questionTypes) {
                Integer index = mappingConfig.getQuestionTypeMap().get(type);
                if (index != null && index >= 0 && index < 5) {
                    weight[index] = 1;
                }
            }
        }
        requestBody.put("weights_set", weight);
        String example = (String) input.getOrDefault("example", "");
        requestBody.put("example", example);
        Integer count = (Integer) input.getOrDefault("count", 100);
        requestBody.put("count", count);
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);

        // 发送请求
        ResponseEntity<String> response = restTemplate.postForEntity(FLASK_URL, requestEntity, String.class);

        List<Map<String, Object>> result;
        try {
            // 解析响应
            String innerJson = objectMapper.readValue(response.getBody(), String.class);
            result = objectMapper.readValue(innerJson, new TypeReference<>() {});
            String str = objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(result);
            System.out.println(str);
            // 插入数据库
            for (Map<String, Object> item : result) {
                DataInfo dataInfo = new DataInfo();
                dataInfo.setDimension(DimensionEnum.valueOf((mappingConfig.getDimensionMap().get(dimensionZh)).toUpperCase()));
                dataInfo.setDataSource(DataSourceEnum.MODEL_GENERATION);
                dataInfo.setModelId(1L);
                dataInfo.setMetricId(metricMapper.selectIdByName(mappingConfig.getMetricMap().get(metricZh)));
                dataInfo.setSubMetricId(subMetricMapper.selectIdByName(mappingConfig.getSubMetricMap().get(subMetricZh)));
                dataInfo.setQuestionType(QuestionTypeEnum.valueOf(((String)item.get("type_index")).toUpperCase()));
                dataInfo.setQuestion(String.valueOf(item.get("question")));
                dataInfo.setAnswer(String.valueOf(item.get("answer")));
                if (item.containsKey("options")) {
                    Object optionsObj = item.get("options");
                    if (optionsObj instanceof List<?>) {
                        List<?> optionsList = (List<?>) optionsObj;
                        String optionsStr = optionsList.stream()
                                .map(Object::toString)
                                .collect(Collectors.joining("|"));
                        dataInfo.setOptions(optionsStr);
                    } else {
                        dataInfo.setOptions(optionsObj.toString());
                    }
                } else {
                    dataInfo.setOptions(null);
                }
                dataInfoMapper.insert(dataInfo);
            }
            return str;
        } catch (JsonProcessingException e) {
            throw new RuntimeException("解析算法返回结果失败", e);
        } catch (Exception e) {
            throw new RuntimeException("解析或保存数据失败", e);
        }


    }
}

