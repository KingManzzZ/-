package com.example.llmtest.utils;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.example.llmtest.exceptionhandler.BusinessException;
import com.example.llmtest.exceptionhandler.ReturnCode;
import com.example.llmtest.mapper.DataInfoMapper;
import com.example.llmtest.mapper.MetricMapper;
import com.example.llmtest.mapper.SubMetricMapper;
import com.example.llmtest.pojo.entity.DataInfo;
import com.example.llmtest.pojo.enums.DimensionEnum;
import com.example.llmtest.pojo.enums.QuestionTypeEnum;
import com.example.llmtest.pojo.vo.DataInfoVO;
import io.micrometer.common.util.StringUtils;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Component
public class TestDataGenerator {
    private final DataInfoMapper dataInfoMapper;
    private final MetricMapper metricMapper;
    private final  SubMetricMapper subMetricMapper;
    private final CustomUtil customUtil;

    public TestDataGenerator(DataInfoMapper dataInfoMapper, MetricMapper metricMapper,
                                 SubMetricMapper subMetricMapper, CustomUtil customUtil) {
        this.dataInfoMapper = dataInfoMapper;
        this.metricMapper = metricMapper;
        this.subMetricMapper = subMetricMapper;
        this.customUtil = customUtil;
    }
    public List<DataInfoVO> generateData(String dimension, String metric, Integer totalQuestions) {
        // 1. 参数校验
        if (StringUtils.isNotBlank(dimension)) {
            if (!customUtil.getDimensionMap().containsValue(dimension)) {
                throw new BusinessException(ReturnCode.RC400.getCode(), "维度不存在");
            }
        }
        if (StringUtils.isNotBlank(metric)) {
            if (!customUtil.getMetricMap().containsValue(metric)) {
                throw new BusinessException(ReturnCode.RC400.getCode(), "指标不存在");
            }
        }
        // 2. 如果是性能维度，必须指定具体指标
        if (DimensionEnum.PERFORMANCE.getValue().equals(dimension)) {
            if (metric == null || !customUtil.getMetricMap().containsValue(metric)) {
                throw new BusinessException(ReturnCode.RC400.getCode(), "性能维度未指定具体指标");
            }
        }
        Long metricId = null;
        if (metric != null) {
            metricId = metricMapper.selectIdByName(metric);
        }

        // 2. 获取各类型题目数量
        Map<QuestionTypeEnum, Integer> typeCountMap = new HashMap<>();
        for (QuestionTypeEnum type : QuestionTypeEnum.values()) {
            QueryWrapper<DataInfo> typeQuery = new QueryWrapper<>();
            typeQuery.eq("dimension", dimension)
                    .eq("question_type", type);
            if (metricId != null) {
                typeQuery.eq("metric_id", metricId);
            }
            typeCountMap.put(type, Math.toIntExact(dataInfoMapper.selectCount(typeQuery)));
        }

        // 3. 新增：获取各指标题目数量
        Map<String, Integer> metricCountMap = new HashMap<>();
        if (DimensionEnum.PERFORMANCE.getValue().equals(dimension)) {
            // 性能维度特殊处理
            if ("complex_reasoning_skill".equals(metric) || "long_text_comprehension_skill".equals(metric)) {
                // 处理子指标
                for (Map.Entry<String, String> entry : customUtil.getSubMetricMap().entrySet()) {
                    QueryWrapper<DataInfo> metricQuery = new QueryWrapper<>();
                    metricQuery.eq("dimension", dimension)
                            .eq("metric_id", metricId)
                            .eq("sub_metric_id", subMetricMapper.selectIdByName(entry.getKey()));
                    metricCountMap.put(entry.getKey(), Math.toIntExact(dataInfoMapper.selectCount(metricQuery)));
                }
            } else {
                // 系统响应效率直接处理
                metricCountMap.put(metric, Math.toIntExact(dataInfoMapper.selectCount(
                        new QueryWrapper<DataInfo>()
                                .eq("dimension", dimension)
                                .eq("metric_id", metricId)
                )));
            }
        } else {
            // 其他维度处理
            for (Map.Entry<String, String> entry : customUtil.getMetricMap().entrySet()) {
                if (customUtil.getDimensionMetrics(dimension).contains(entry.getKey())) {
                    QueryWrapper<DataInfo> metricQuery = new QueryWrapper<>();
                    metricQuery.eq("dimension", dimension)
                            .eq("metric_id", metricMapper.selectIdByName(entry.getKey()));
                    metricCountMap.put(entry.getKey(), Math.toIntExact(dataInfoMapper.selectCount(metricQuery)));
                }
            }
        }

        // 4. 计算每种类型和指标应分配的题目数
        Map<QuestionTypeEnum, Integer> allocatedTypes = new HashMap<>();
        Map<String, Integer> allocatedMetrics = new HashMap<>();

        // 计算总可用题目数（题型和指标）
        int totalAvailableByType = typeCountMap.values().stream().mapToInt(Integer::intValue).sum();
        int totalAvailableByMetric = metricCountMap.values().stream().mapToInt(Integer::intValue).sum();

        // 按比例分配题目（题型和指标）
        for (Map.Entry<QuestionTypeEnum, Integer> entry : typeCountMap.entrySet()) {
            if (entry.getValue() > 0) {
                double ratio = (double) entry.getValue() / totalAvailableByType;
                int allocated = Math.max(1, (int) Math.round(ratio * totalQuestions));
                allocatedTypes.put(entry.getKey(), allocated);
            }
        }

        for (Map.Entry<String, Integer> entry : metricCountMap.entrySet()) {
            if (entry.getValue() > 0) {
                double ratio = (double) entry.getValue() / totalAvailableByMetric;
                int allocated = Math.max(1, (int) Math.round(ratio * totalQuestions));
                allocatedMetrics.put(entry.getKey(), allocated);
            }
        }

        // 5. 随机抽取题目（同时考虑题型和指标）
        List<DataInfo> questionList = new ArrayList<>();
        for (Map.Entry<QuestionTypeEnum, Integer> typeEntry : allocatedTypes.entrySet()) {
            for (Map.Entry<String, Integer> metricEntry : allocatedMetrics.entrySet()) {
                QueryWrapper<DataInfo> typeQuery = new QueryWrapper<>();
                typeQuery.select("data_id", "question", "options", "answer", "dimension",
                                "question_type", "data_source", "update_time", "transformation_type",
                                "transformation_description", "original_data_id")
                        .eq("dimension", dimension)
                        .eq("question_type", typeEntry.getKey());

                if (DimensionEnum.PERFORMANCE.getValue().equals(dimension)) {
                    if ("complex_reasoning_skill".equals(metric) || "long_text_comprehension_skill".equals(metric)) {
                        typeQuery.eq("sub_metric_id", subMetricMapper.selectIdByName(metricEntry.getKey()));
                    } else {
                        typeQuery.eq("metric_id", metricId);
                    }
                } else {
                    typeQuery.eq("metric_id", metricMapper.selectIdByName(metricEntry.getKey()));
                }

                // 计算该组合应抽取的题目数
                int questionsToSelect = Math.min(
                        //将当前题型的题目平均分配给所有指标
                        typeEntry.getValue() / allocatedMetrics.size(),
                        //将当前指标的题目平均分配给所有题型
                        metricEntry.getValue() / allocatedTypes.size()
                );

                if (questionsToSelect > 0) {
                    typeQuery.last("ORDER BY RAND() LIMIT " + questionsToSelect);
                    questionList.addAll(dataInfoMapper.selectList(typeQuery));
                }
            }
        }

        return questionList.stream()
                .map(customUtil::convertToDataInfoVO)
                .collect(Collectors.toList());
    }
}
