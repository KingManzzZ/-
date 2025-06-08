package com.example.llmtest.algorithm;

import com.example.llmtest.mapper.DataInfoMapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.ResponseEntity;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;


public class CallBService {


    public static void main(String[] args) {

        /*RestTemplate restTemplate = new RestTemplate();
        String bUrl = "https://test-1-www.u659522.nyat.app:40794/evolve"; // 确保使用 HTTPS


        // 设置请求头
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);


        // 构造请求参数（符合算法端要求的格式）
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("domain", "安全");
        requestBody.put("weights_set", new int[]{0, 0, 0, 1, 0});
        requestBody.put("example", "示例内容");
        requestBody.put("count", 5);


        // 构造请求实体（包含 body 和 header）
        HttpEntity<Map<String, Object>> requestEntity = new HttpEntity<>(requestBody,headers);
        // 发送 POST 请求
        ResponseEntity<String> response = restTemplate.postForEntity(bUrl, requestBody, String.class);

        String rawResponse = response.getBody();

        // 创建 ObjectMapper 实例
        ObjectMapper objectMapper = new ObjectMapper();

        try {
            // 第一步：把原始字符串解析为内部 JSON 字符串（去掉一层转义）
            String innerJson = objectMapper.readValue(rawResponse, String.class);

            // 第二步：将内部 JSON 字符串解析为 Map 列表
            List<Map<String, Object>> result = objectMapper.readValue(
                    innerJson,
                    new TypeReference<List<Map<String, Object>>>() {}
            );

            // 第三步：重新序列化为格式良好的 JSON 字符串（Unicode 会被自动转换为中文）
            String prettyPrintJson = objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(result);

            System.out.println(prettyPrintJson);

        } catch (Exception e) {
            System.err.println("JSON 处理失败：" + e.getMessage());
            e.printStackTrace();
        }*/

        HashMap<String, Object> map = new HashMap<>();
        map.put("options", "[ \"A. 15 km\", \"B. 20 km\", \"C. 25 km\", \"D. 30 km\" ]");
        Object optionsObj =map.get("options");
        System.out.println(optionsObj);
        if (optionsObj instanceof List<?>) {
            List<?> optionsList = (List<?>) optionsObj;
            // 拼接所有选项，用 "|" 分隔
            String optionsStr = "\"" + String.join("\"|\"", optionsList.stream()
                    .map(Object::toString)
                    .collect(Collectors.toList())) + "\"";
            System.out.println(optionsStr);
        }
    }
}

