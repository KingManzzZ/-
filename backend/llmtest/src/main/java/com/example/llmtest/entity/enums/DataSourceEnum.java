package com.example.llmtest.entity.enums;

import com.baomidou.mybatisplus.annotation.EnumValue;
import com.baomidou.mybatisplus.annotation.IEnum;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Arrays;


public enum DataSourceEnum implements IEnum<String> {
    INPUT("input"),
    CRAWLER("crawler"),
    MODEL_GENERATION("model_generation");

    @EnumValue
    @JsonValue
    private final String value;

    DataSourceEnum(String value) {this.value = value;}
    @Override
    public String getValue() {return this.value;}

    //  JSON 反序列化时使用
    @JsonCreator
    public static DataSourceEnum forValue(String value) {
        return Arrays.stream(values())
                .filter(e -> e.value.equals(value))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("无效值: " + value));
    }
}
