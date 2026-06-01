package com.example.llmtest.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.llmtest.pojo.entity.TestQuestions;
import com.example.llmtest.pojo.vo.TestExactVO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface TestQuestionsMapper extends BaseMapper<TestQuestions> {

    int insertBatch(List<TestQuestions> list);

    int insertBatchWithoutScore(List<TestQuestions> list);

    @Select("SELECT d.data_id, d.question, d.answer, d.options,  tq.model_output, " +
            "CASE WHEN ts.item_type = 'question' THEN ts.score ELSE NULL END AS score " +
            "FROM test_questions tq " +
            "INNER JOIN data_info d ON tq.data_id = d.data_id " +
            "LEFT JOIN test_scores ts ON ts.test_id = tq.test_id AND ts.item_id = tq.data_id " +
            "WHERE tq.test_id = #{testId} " +
            "ORDER BY tq.data_id")
    List<TestExactVO> findByTestId(@Param("testId") Long testId);

}
