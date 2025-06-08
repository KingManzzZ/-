package com.example.llmtest.entity.enums;

import com.baomidou.mybatisplus.annotation.EnumValue;
import com.baomidou.mybatisplus.annotation.IEnum;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Arrays;


public enum QuestionTypeEnum implements IEnum<String> {
    CHOICE("choice"),
    JUDGMENT("judgment"),
    SHORT_ANSWER("short_answer"),
    ONLY_QUESTION("only_question"),
    COMPARE_QUESTION("compare_question");

    @EnumValue
    @JsonValue
    private final String value;

    QuestionTypeEnum(String value) {
        this.value = value;
    }
    @Override
    public String getValue() {return this.value;}

    //  JSON 反序列化时使用
    @JsonCreator
    public static QuestionTypeEnum forValue(String value) {
        return Arrays.stream(values())
                .filter(e -> e.value.equals(value))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("无效值: " + value));
    }

    public static boolean contains(String value) {
        for (QuestionTypeEnum type : QuestionTypeEnum.values()) {
            if (type.value.equals(value)) {
                return true;
            }
        }
        return false;
    }
}
