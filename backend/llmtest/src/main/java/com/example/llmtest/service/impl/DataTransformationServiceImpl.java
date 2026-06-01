package com.example.llmtest.service.impl;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.llmtest.exceptionhandler.BusinessException;
import com.example.llmtest.exceptionhandler.ReturnCode;
import com.example.llmtest.mapper.DataInfoMapper;
import com.example.llmtest.mapper.ModelMapper;
import com.example.llmtest.pojo.dto.TransformationDTO;
import com.example.llmtest.pojo.entity.DataInfo;
import com.example.llmtest.pojo.enums.DataSourceEnum;
import com.example.llmtest.pojo.enums.TransformationTypeEnum;
import com.example.llmtest.pojo.vo.DataInfoVO;
import com.example.llmtest.service.DataTransformationService;
import com.example.llmtest.utils.CustomUtil;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.BeanUtils;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;

import java.sql.Timestamp;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Slf4j
@Service
public class DataTransformationServiceImpl extends ServiceImpl<DataInfoMapper, DataInfo> implements DataTransformationService {

    private static final String FLASK_URL = "https://www.u659522.nyat.app:26249/evolve";
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final RestTemplate restTemplate;
    private final CustomUtil customUtil;
    private final ModelMapper modelMapper;
    private final DataInfoMapper dataInfoMapper;

    public DataTransformationServiceImpl(RestTemplate restTemplate, CustomUtil customUtil,
                                         ModelMapper modelMapper, DataInfoMapper dataInfoMapper) {
        this.restTemplate = restTemplate;
        this.customUtil = customUtil;
        this.modelMapper = modelMapper;
        this.dataInfoMapper = dataInfoMapper;
    }

    /**
     * 题目变形
     * @param dto
     * @return
     */
    @Override
    @Transactional
    public List<DataInfoVO> transformByModel(TransformationDTO dto) {
        return transformBatch(dto);
    }


    private List<DataInfoVO> transformBatch(TransformationDTO dto) {
        List<Long> dataIds = dto.getDataIds();
        String transformationType = dto.getTransformationType();
        if (StringUtils.isNotBlank(transformationType)) {
            if (!customUtil.getTransformationTypeMap().containsValue(transformationType)) {
                throw new BusinessException(ReturnCode.RC400.getCode(), "变形方法不存在");
            }
        }

        List<Map<String, Object>> requestBody = new ArrayList<>();
        List<DataInfo> originalDataList = new ArrayList<>();

        // 构造批量请求体
        for (int i = 0; i < dataIds.size(); i++) {
            Long dataId = dataIds.get(i);
            DataInfo dataInfo = dataInfoMapper.selectById(dataId);
            originalDataList.add(dataInfo);

            HashMap<String, Object> content = new HashMap<>();
            content.put("rowIdx", i);
            content.put("question", dataInfo.getQuestion());
            String optionsString = dataInfo.getOptions();
            String[] options = customUtil.parseStringToArray(optionsString);
            content.put("options", options);

            content.put("answer", dataInfo.getAnswer());
            content.put("type_index", dataInfo.getQuestionType().getValue());

            HashMap<String, Object> questionObj = new HashMap<>();
            questionObj.put("question0", content);
            questionObj.put("evolve_type", transformationType);

            requestBody.add(questionObj);
        }
        // 构造请求头
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<List<Map<String, Object>>> requestEntity = new HttpEntity<>(requestBody, headers);

        // 发送批量请求
        LocalDateTime startTime = LocalDateTime.now();
        ResponseEntity<String> response = restTemplate.postForEntity(FLASK_URL, requestEntity, String.class);
        if (!response.getStatusCode().is2xxSuccessful()) {
            throw new BusinessException(500, "调用 Flask 接口失败，状态码：" + response.getStatusCode());
        }
        try {
            List<Map<String, Object>> allResults = objectMapper.readValue(response.getBody(), new TypeReference<>() {});

            List<DataInfo> resultDataList = new ArrayList<>();

            // 处理每个返回的结果
            for (int i = 0; i < allResults.size(); i++) {
                Map<String, Object> result = allResults.get(i);
                DataInfo originalData = originalDataList.get(i);
                if (originalData != null) log.info("第{}道题变形成功", i + 1);
                // 构造新题目数据
                DataInfo newData = new DataInfo();
                BeanUtils.copyProperties(originalData, newData);
                newData.setDataId(null);
                newData.setUpdateTime(new Timestamp(System.currentTimeMillis()));
                newData.setQuestion(String.valueOf(result.get("question")));
                newData.setAnswer(String.valueOf(result.get("answer")));
                if (result.containsKey("options")) {
                    Object optionsObj = result.get("options");
                    String optionsString = customUtil.parseArrayToString(optionsObj);
                    newData.setOptions(optionsString);
                } else {
                    newData.setOptions(null);
                }
                newData.setModelId(modelMapper.selectIdByName((String)result.get("model")));
                newData.setDataSource(DataSourceEnum.MODEL_GENERATION);
                newData.setIsTransformed(1);
                newData.setOriginalDataId(originalData.getDataId());
                newData.setTransformationType(TransformationTypeEnum.valueOf(transformationType.toUpperCase()));
                newData.setTransformationDescription(String.valueOf(result.get("process")));

                dataInfoMapper.insert(newData);
                resultDataList.add(newData);
            }

            LocalDateTime endTime = LocalDateTime.now();
            Duration duration = Duration.between(startTime, endTime);
            log.info("批量处理{}道题目总耗时: {}分{}秒", dataIds.size(), duration.toMinutes(), duration.getSeconds() % 60);

            return resultDataList.stream()
                    .map(customUtil::convertToDataInfoVO)
                    .collect(Collectors.toList());
        } catch (JsonProcessingException e) {
            throw new RuntimeException("解析算法返回结果失败", e);
        } catch (Exception e) {
            throw new RuntimeException("解析或保存数据失败", e);
        }
    }
}
