package com.example.llmtest.service.impl;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.llmtest.exceptionhandler.BusinessException;
import com.example.llmtest.exceptionhandler.ReturnCode;
import com.example.llmtest.mapper.*;
import com.example.llmtest.pojo.dto.TestDTO;
import com.example.llmtest.pojo.entity.DataInfo;
import com.example.llmtest.pojo.entity.TestInfo;
import com.example.llmtest.pojo.entity.TestQuestions;
import com.example.llmtest.pojo.entity.TestScores;
import com.example.llmtest.pojo.enums.DimensionEnum;
import com.example.llmtest.pojo.vo.TestExactVO;
import com.example.llmtest.pojo.vo.TestResultVO;
import com.example.llmtest.pojo.vo.TestInfoVO;
import com.example.llmtest.service.TestService;
import com.example.llmtest.utils.CustomUtil;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Service
@Slf4j
public class TestServiceImpl extends ServiceImpl<TestInfoMapper, TestInfo> implements TestService {

    private static final String FLASK_URL = "https://www.u659522.nyat.app:26249/evaluation/general_process";
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final RestTemplate restTemplate;
    private final CustomUtil customUtil;
    private final DataInfoMapper dataInfoMapper;
    private final ModelMapper modelMapper;
    private final MetricMapper metricMapper;
    private final SubMetricMapper subMetricMapper;
    private final TestScoresMapper testScoresMapper;
    private final TestQuestionsMapper testQuestionsMapper;
    private final TestInfoMapper testInfoMapper;

    public TestServiceImpl(RestTemplate restTemplate, CustomUtil customUtil,
                           DataInfoMapper dataInfoMapper, ModelMapper modelMapper,
                           MetricMapper metricMapper,
                           TestQuestionsMapper testQuestionsMapper,
                           SubMetricMapper subMetricMapper, TestScoresMapper testScoresMapper, TestInfoMapper testInfoMapper) {
        this.restTemplate = restTemplate;
        this.customUtil = customUtil;
        this.dataInfoMapper = dataInfoMapper;
        this.modelMapper = modelMapper;
        this.metricMapper = metricMapper;
        this.subMetricMapper = subMetricMapper;
        this.testQuestionsMapper = testQuestionsMapper;
        this.testScoresMapper = testScoresMapper;
        this.testInfoMapper = testInfoMapper;
    }

    /**
     * 系统响应效率和公平性测试，按题目给分
     * @return vo
     */
    @Override
    @Transactional
    public TestResultVO questionTest(TestDTO dto) {
        String dimension = dto.getDimension();//维度只能是performance 或者fairness
        String metric = dto.getMetric();//metric只能是system_responsiveness
        if (StringUtils.isBlank(dimension) || (!"performance".equals(dimension) && !"fairness".equals(dimension))) {
            throw new BusinessException(ReturnCode.RC400.getCode(), "维度应为性能或者公平性");
        }
        if (StringUtils.isNotBlank(metric) && !"system_responsiveness".equals(metric)) {
            throw new BusinessException(ReturnCode.RC400.getCode(), "指标应为系统响应效率");
        }
        boolean flag = "performance".equals(dimension);//true表示性能，false表示公平性
        //构建请求体
        HashMap<String, Object> requestBody = new HashMap<>();
        requestBody.put("model", dto.getModelName());
        if (!flag) {
            requestBody.put("domain", dimension);
        } else {
            requestBody.put("domain", metric);
        }
        //判断每个metric下的各小指标比例是否正常（10%~90%）
        isValid(dto.getQuestionList(), dimension, metric);

        List<Long> questionList = dto.getQuestionList();
        int questionCount = questionList.size();
        List<Map<String, Object>> qsList = new ArrayList<>();
        for (Long dataId : questionList) {
            DataInfo data = dataInfoMapper.selectById(dataId);
            if (StringUtils.isNotBlank(dimension) && !data.getDimension().getValue().equals(dimension)) {
                log.warn("id为{}的题目维度不符合，已自动跳过", dataId);
                questionCount--;
                continue;
            }
            if (StringUtils.isNotBlank(metric) && !metricMapper.selectIdByName(metric).equals(data.getMetricId())) {
                log.warn("id为{}的题目指标不符合，已自动跳过", dataId);
                questionCount--;
                continue;
            }
            HashMap<String, Object> everyData = new HashMap<>();
            everyData.put("rowIdx", dataId);
            //公平性题目进行特殊处理
            /*
            "question": {
                "A": "ass",
                "B": "hole"
            }
             */
            if (data.getQuestionType().getValue().equals("compare_question")) {
                Map<String, String> compareQuestion = new HashMap<>();
                String[] options = data.getQuestion().split("\\|");
                for (String option : options) {
                    compareQuestion.put(String.valueOf(option.charAt(0)), option.substring(2));
                }
                everyData.put("question", compareQuestion);
            } else {
                everyData.put("question", data.getQuestion());
            }
            everyData.put("answer", data.getAnswer());
            everyData.put("type_index", data.getQuestionType().getValue());
            String[] options = customUtil.parseStringToArray(data.getOptions());
            everyData.put("options", options);
            //判断min_metric内容
            if (flag) {
                everyData.put("min_metric", "system_responsiveness");
            } else {
                everyData.put("min_metric", metricMapper.selectNameById(data.getMetricId()));
            }
            qsList.add(everyData);
        }
        requestBody.put("qs_list", qsList);
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        log.info("test1接口发送请求");
        LocalDateTime startTime = LocalDateTime.now();
        HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);
  
        //接受响应
        ResponseEntity<String> response = restTemplate.postForEntity(FLASK_URL, requestEntity, String.class);

        log.info("接收响应，状态码: {}, 耗时: {}秒",
                response.getStatusCode(),
                Duration.between(startTime, LocalDateTime.now()).getSeconds());
        if (!response.getStatusCode().is2xxSuccessful()) {
            throw new BusinessException(500, "调用 Flask 接口失败，状态码：" + response.getStatusCode());
        }

        TestResultVO vo = new TestResultVO();
        try {
            JsonNode jsonNode = objectMapper.readTree(response.getBody());
            List<Map<String, Object>> modelResponse = Optional.ofNullable(jsonNode.get("model_response"))
                    .filter(JsonNode::isArray)
                    .map(node -> objectMapper.convertValue(node, new TypeReference<List<Map<String, Object>>>() {}))
                    .orElseThrow(() -> new BusinessException(500, "模型响应格式不正确，model_response应为List类型"));

            Map<String, Object> score = Optional.ofNullable(jsonNode.get("score"))
                    .filter(JsonNode::isObject)
                    .map(node -> objectMapper.convertValue(node, new TypeReference<Map<String, Object>>() {}))
                    .orElseThrow(() -> new BusinessException(500, "模型响应格式不正确，score应为Map类型"));

            Double finalScore = (Double)score.get("final_score");
            vo.setFinalScore(finalScore);
            List<Double> arrayList = (List<Double>) score.get("single_score");
            Double[] singleScore = arrayList.toArray(Double[]::new);

            vo.setSingleScore(singleScore);
            String resultDescription = "finalScore is " + finalScore + "singleScore is " + Arrays.toString(singleScore);
            log.info(resultDescription);
            TestInfo testInfo = TestInfo.builder()
                    .name(dto.getTestName())
                    .modelId(modelMapper.selectIdByName(dto.getModelName()))
                    .dimension(DimensionEnum.forValue(dimension))
                    .metricId(metricMapper.selectIdByName(dto.getMetric()))
                    .os(dto.getOs())
                    .cpu(dto.getCpu())
                    .gpu(dto.getGpu())
                    .finalScore(finalScore)
                    .count((long)questionCount)
                    .resultDescription(resultDescription)
                    .build();
            this.save(testInfo);
            vo.setTestId(testInfo.getTestId());

            //插入数据至test_questions表和test_scores表
            List<TestQuestions> testQuestionsList = new ArrayList<>();
            List<TestScores> testScoresList = new ArrayList<>();
            int i = 0;
            for (Map<String, Object> responseItem : modelResponse) {
                Long testId = testInfo.getTestId();
                Long dataId = ((Integer)responseItem.get("dataId")).longValue();
                TestQuestions testQuestion = new TestQuestions();
                testQuestion.setTestId(testId);
                testQuestion.setDataId(dataId);
                //问题比较组的模型输出进行特殊处理
                DataInfo data = dataInfoMapper.selectById(dataId);
                if (data.getQuestionType().getValue().equals("compare_question")) {
                    String str = responseItem.get("response_A") + "|" +
                            responseItem.get("response_B");
                    testQuestion.setModelOutput(str);
                } else {
                    testQuestion.setModelOutput(String.valueOf(responseItem.get("response")));
                }
                testQuestionsList.add(testQuestion);

                TestScores testScore = new TestScores();
                testScore.setTestId(testId);
                testScore.setDimension(DimensionEnum.forValue(dimension));
                testScore.setItemType("question");
                testScore.setItemId(dataId);
                //TODO: singleScore分数可能有缺失，没有考虑到
                if (i < singleScore.length)testScore.setScore(singleScore[i]);
                i++;
                testScoresList.add(testScore);

            }
            int count1 = testQuestionsMapper.insertBatch(testQuestionsList);
            log.info("成功向test_questions表中插入{}条数据", count1);
            int count2 = testScoresMapper.insertBatch(testScoresList);
            log.info("成功向test_scores表中插入{}条数据", count2);

            customUtil.timeSpent(startTime, LocalDateTime.now(), (long)questionList.size());
        } catch (JsonProcessingException e) {
            throw new RuntimeException("JSON processing error", e);
        } catch (Exception e) {
            log.error(e.getMessage());
            throw new RuntimeException(e);
        }
        return vo;
    }

    /**
     * 剩余指标测试
     * @return vo
     */
    @Override
    @Transactional
    public TestResultVO metricTest(TestDTO dto) {
        String dimension = dto.getDimension();
        String metric = dto.getMetric();
        boolean flag = "performance".equals(dimension);//true表示复杂推理能力或者长文本理解能力，false表示其他两个维度
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
        //判断每个metric下的各小指标比例是否正常（10%~90%）
        isValid(dto.getQuestionList(), dimension, metric);

        //构建请求体
        HashMap<String, Object> requestBody = new HashMap<>();
        requestBody.put("model", dto.getModelName());
        if (flag) {
            requestBody.put("domain", metric);
        } else {
            requestBody.put("domain", dimension);
        }
        List<Long> questionList = dto.getQuestionList();
        int questionCount = questionList.size();
        List<Map<String, Object>> qsList = new ArrayList<>();
        for (Long dataId : questionList) {
            DataInfo data = dataInfoMapper.selectById(dataId);
            if (StringUtils.isNotBlank(dimension) && !data.getDimension().getValue().equals(dimension)) {
                log.warn("id为{}的题目维度不符合，已自动跳过", dataId);
                questionCount--;
                continue;
            }
            if (StringUtils.isNotBlank(metric) && !metricMapper.selectIdByName(metric).equals(data.getMetricId())) {
                log.warn("id为{}的题目指标不符合，已自动跳过", dataId);
                questionCount--;
                continue;
            }
            HashMap<String, Object> everyData = new HashMap<>();
            everyData.put("rowIdx", dataId);
            everyData.put("question", data.getQuestion());
            everyData.put("answer", data.getAnswer());
            everyData.put("type_index", data.getQuestionType().getValue());
            String[] options = customUtil.parseStringToArray(data.getOptions());
            everyData.put("options", options);

            //判断min_metric内容
            if (flag) {
                everyData.put("min_metric", subMetricMapper.selectNameById(data.getSubMetricId()));
            } else {
                everyData.put("min_metric", metricMapper.selectNameById(data.getMetricId()));
            }
            qsList.add(everyData);
        }
        requestBody.put("qs_list", qsList);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        log.info("test2接口发送请求");
        LocalDateTime startTime = LocalDateTime.now();
        HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody, headers);

        //接受响应
        ResponseEntity<String> response = restTemplate.postForEntity(FLASK_URL, requestEntity, String.class);
        log.info("接收响应，状态码: {}, 耗时: {}秒",
                response.getStatusCode(),
                Duration.between(startTime, LocalDateTime.now()).getSeconds());
        if (!response.getStatusCode().is2xxSuccessful()) {
            throw new BusinessException(500, "调用 Flask 接口失败，状态码：" + response.getStatusCode());
        }

        TestResultVO vo = new TestResultVO();
        try {
            JsonNode jsonNode = objectMapper.readTree(response.getBody());
            List<Map<String, Object>> modelResponse = Optional.ofNullable(jsonNode.get("model_response"))
                    .filter(JsonNode::isArray)
                    .map(node -> objectMapper.convertValue(node, new TypeReference<List<Map<String, Object>>>() {}))
                    .orElseThrow(() -> new BusinessException(500, "模型响应格式不正确，model_response应为List类型"));

            Map<String, Object> score = Optional.ofNullable(jsonNode.get("score"))
                    .filter(JsonNode::isObject)
                    .map(node -> objectMapper.convertValue(node, new TypeReference<Map<String, Object>>() {}))
                    .orElseThrow(() -> new BusinessException(500, "模型响应格式不正确，score应为Map类型"));
            Double finalScore = (double)score.get("final_score");
            vo.setFinalScore(finalScore);
            StringBuilder resultDescription = new StringBuilder();
            resultDescription.append("测试总分为: ").append(finalScore).append("各指标测试分数为: ");
            //获取各指标分数
            Map<String, Double> metricScores = new HashMap<>();
            score.entrySet().stream()
                    .filter(entry -> !entry.getKey().equals("final_score"))
                    .forEach(entry -> {
                        String key = entry.getKey();
                        Double value = (Double)entry.getValue();
                        metricScores.put(key, value);
                    });
            vo.setMetricScores(metricScores);

            metricScores.forEach((key, value) ->
                    resultDescription.append(key).append(": ").append(value).append(" "));
            log.info(resultDescription.toString());


            TestInfo testInfo = TestInfo.builder()
                    .name(dto.getTestName())
                    .modelId(modelMapper.selectIdByName(dto.getModelName()))
                    .dimension(DimensionEnum.valueOf(dimension.toUpperCase()))
                    .metricId(metricMapper.selectIdByName(dto.getMetric()))
                    .os(dto.getOs())
                    .cpu(dto.getCpu())
                    .gpu(dto.getGpu())
                    .finalScore(finalScore)
                    .count((long)questionCount)
                    .resultDescription(resultDescription.toString())
                    .build();
            this.save(testInfo);
            vo.setTestId(testInfo.getTestId());
            //插入数据至test_questions表
            List<TestQuestions> testQuestionsList = new ArrayList<>();
            for (Map<String, Object> responseItem : modelResponse) {
                Long testId = testInfo.getTestId();
                Long dataId = ((Integer)responseItem.get("dataId")).longValue();
                TestQuestions testQuestion = new TestQuestions();
                testQuestion.setTestId(testId);
                testQuestion.setDataId(dataId);
                testQuestion.setModelOutput(String.valueOf(responseItem.get("response")));
                testQuestionsList.add(testQuestion);
            }
            int count1 = testQuestionsMapper.insertBatch(testQuestionsList);
            log.info("成功向test_questions表中插入{}条数据", count1);

            //插入数据至test_scores表
            List<TestScores> testScoresList = metricScores.entrySet().stream()
                    .map(entry -> {
                        TestScores testScore = new TestScores();
                        testScore.setTestId(testInfo.getTestId());
                        testScore.setItemType("metric");
                        testScore.setDimension(DimensionEnum.forValue(dimension));
                        testScore.setItemId(flag ?
                                subMetricMapper.selectIdByName(entry.getKey()) :
                                metricMapper.selectIdByName(entry.getKey()));
                        testScore.setScore(entry.getValue());
                        return testScore;
                    })
                    .collect(Collectors.toList());
            int count2 = testScoresMapper.insertBatch(testScoresList);
            log.info("成功向test_scores表中插入{}条数据", count2);

            customUtil.timeSpent(startTime, LocalDateTime.now(), (long)questionList.size());
        } catch (JsonProcessingException e) {
            throw new RuntimeException("JSON processing error", e);
        } catch (Exception e) {
            log.error(e.getMessage());
            throw new RuntimeException(e);
        }
        return vo;
    }

    /**
     * 根据选择的题目判断每种指标所占比例是否合适
     *
     * @param questionList
     */
    //公平性测试数据 gender16432, race16732, age17132, religion17432, politics17832
    public void isValid(List<Long> questionList, String dimension, String metric) {
        HashMap<Long, Integer> cnt = new HashMap<>();
        int total = questionList.size();
        if (total <= 1) return;
        //非性能
        if (!dimension.equals(DimensionEnum.PERFORMANCE.getValue())) {
            for (Long dataId : questionList) {
                DataInfo data = dataInfoMapper.selectById(dataId);
                Long metricId = data.getMetricId();
                cnt.merge(metricId, 1, Integer::sum);
            }
            if (dimension.equals("reliability") && cnt.size() < 4) {
                log.error("可靠性指标数量不足");
                throw new BusinessException(ReturnCode.RC400.getCode(), "可靠性指标数量不足");
            }
            if (dimension.equals("safety") && cnt.size() < 8) {
                log.error("安全性指标数量不足");
                throw new BusinessException(ReturnCode.RC400.getCode(), "安全性指标数量不足");
            }
            if (dimension.equals("fairness") && cnt.size() < 5) {
                log.error("公平性指标数量不足");
                throw new BusinessException(ReturnCode.RC400.getCode(), "公平性指标数量不足");
            }
            for (Map.Entry<Long, Integer> entry : cnt.entrySet()) {
                double percentage = (double) entry.getValue() / total * 100;
                if (percentage < 10 || percentage > 90) {
                    log.error("当前测试维度为 {}，指标 {} 的题目数量占比 {}%，不在合理范围内(10%-90%)", dimension, entry.getKey(), percentage);
                    throw new BusinessException(ReturnCode.RC400.getCode(),
                            String.format("当前测试维度为 %s，指标 %s 的题目数量占比 %.2f%%，不在合理范围内(10%%-90%%)",
                                    dimension, entry.getKey(), percentage));
                }
            }
        } else if (metric.equals("complex_reasoning_skill") || metric.equals("long_text_comprehension_skill")) {
            for (Long dataId : questionList) {
                DataInfo data = dataInfoMapper.selectById(dataId);
                Long subMetricId = data.getSubMetricId();
                cnt.merge(subMetricId, 1, Integer::sum);
            }
            if (cnt.size() < 3) {
                log.error("子指标数量不足");
                throw new BusinessException(ReturnCode.RC400.getCode(), "子指标数量不足");
            }
            for (Map.Entry<Long, Integer> entry : cnt.entrySet()) {
                double percentage = (double) entry.getValue() / total * 100;
                if (percentage < 10 || percentage > 90) {
                    log.error("当前测试维度为 {}，指标 {} 的题目数量占比 {}%，不在合理范围内(10%-90%)", dimension, entry.getKey(), percentage);
                    throw new BusinessException(ReturnCode.RC400.getCode(),
                            String.format("当前测试维度为 %s，子指标 %s 的题目数量占比 %.2f%%，不在合理范围内(10%%-90%%)",
                                    metric, entry.getKey(), percentage));
                }
            }
        }
        log.info("指标全覆盖且比例正常");
    }


    @Override
    public IPage<TestInfoVO> selectRoughByPage(Long pageNum) {


        Page<TestInfo> page = new Page<>(pageNum, 15);
        IPage<TestInfo> testInfoPage = this.page(page);
        IPage<TestInfoVO> testInfoVOPage = testInfoPage.convert(customUtil::convertToTestInfoVO);
        return testInfoVOPage;
    }

    /**
     * 根据testId查询某一次测试的详细回答
     * @param testId
     * @return
     */
    @Override
    public List<TestExactVO> selectExactByList(Long testId) {
        List<TestExactVO> resList = testQuestionsMapper.findByTestId(testId);
        return resList;
    }
}
