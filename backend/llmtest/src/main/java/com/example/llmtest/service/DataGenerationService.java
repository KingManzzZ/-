package com.example.llmtest.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.example.llmtest.pojo.dto.GenerationByHandDTO;
import com.example.llmtest.pojo.dto.GenerationByModelDTO;
import com.example.llmtest.pojo.entity.DataInfo;
import com.example.llmtest.pojo.vo.DataInfoVO;

import java.util.List;

public interface DataGenerationService extends IService<DataInfo> {
    /**
     * 模型智能生成
     * @param dto
     * @return
     */
    List<DataInfoVO> generateByModel(GenerationByModelDTO dto);

    /**
     * 手动录入
     * @param dto
     * @return
     */
    Boolean generateByHand(GenerationByHandDTO dto);
}
