package com.example.llmtest.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.example.llmtest.pojo.dto.DataInfoPageQueryDTO;
import com.example.llmtest.pojo.entity.DataInfo;
import com.example.llmtest.pojo.vo.DataInfoVO;

public interface DataInfoService extends IService<DataInfo> {

    /**
     * 根据条件分页查询题目
     *
     * @param queryDTO
     * @return
     */
    IPage<DataInfoVO> getDataInfoByConditions(DataInfoPageQueryDTO queryDTO);

    /**
     * 根据题目id查询题目
     * @param dataId
     * @return
     */
    DataInfoVO getDataInfoByDataId(Long dataId);


    boolean updateDataInfo(DataInfo dataInfo);




}
