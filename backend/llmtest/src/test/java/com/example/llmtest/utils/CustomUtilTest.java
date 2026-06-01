package com.example.llmtest.utils;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class CustomUtilTest {

    @Test
    void mapsCausalReasoningToCanonicalSubMetricName() {
        CustomUtil customUtil = new CustomUtil(null, null, null, null, null);

        assertThat(customUtil.getSubMetricMap().get("因果推理"))
                .isEqualTo("causal_reasoning");
        assertThat(customUtil.getSubMetricMap())
                .doesNotContainValue("casual_reasoning");
    }
}
