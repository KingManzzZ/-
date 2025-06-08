package com.example.llmtest.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.service.IService;
import com.example.llmtest.entity.DataInfo;

public interface DataInfoService extends IService<DataInfo> {
    IPage<DataInfo> getDataInfoByPage(int pageNum);
    IPage<DataInfo> getDataInfoByQuestionTypeByPage(int pageNum, String questionType);
    IPage<DataInfo> getDataInfoByPartContentByPage(int pageNum, String dimension, String metric);
    IPage<DataInfo> getDataInfoByAllContentByPage(int pageNum, String dimension, String metric, String subMetric);
    DataInfo getDataInfoByDataId(Long dataId);
    boolean updateDataInfo(DataInfo dataInfo);

    boolean deleteDataInfoByDataId(Long dataId);
}
