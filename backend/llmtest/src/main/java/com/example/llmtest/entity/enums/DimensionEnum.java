package com.example.llmtest.entity.enums;

import com.baomidou.mybatisplus.annotation.EnumValue;
import com.baomidou.mybatisplus.annotation.IEnum;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Arrays;


public enum DimensionEnum implements IEnum<String> {
    PERFORMANCE("performance"),
    RELIABILITY("reliability"),
    SAFETY("safety"),
    FAIRNESS("fairness");

    @EnumValue
    @JsonValue
    private final String value;

    DimensionEnum(String value) {this.value = value;}
    @Override
    public String getValue() {return value;}

    //  JSON 反序列化时使用
    @JsonCreator
    public static DimensionEnum forValue(String value) {
        return Arrays.stream(values())
                .filter(e -> e.value.equals(value))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("无效值: " + value));
    }

    public static boolean contains(String value) {
        for (DimensionEnum type : DimensionEnum.values()) {
            if (type.value.equals(value)) {
                return true;
            }
        }
        return false;
    }
}
