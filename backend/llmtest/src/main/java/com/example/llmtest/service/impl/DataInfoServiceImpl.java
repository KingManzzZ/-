package com.example.llmtest.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.UpdateWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.llmtest.entity.DataInfo;
import com.example.llmtest.mapper.DataInfoMapper;
import com.example.llmtest.mapper.MetricMapper;
import com.example.llmtest.mapper.SubMetricMapper;
import com.example.llmtest.service.DataInfoService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import static com.baomidou.mybatisplus.extension.toolkit.SimpleQuery.list;


@Service
public class DataInfoServiceImpl extends ServiceImpl<DataInfoMapper, DataInfo> implements DataInfoService {

    private final DataInfoMapper dataInfoMapper;
    private final MetricMapper metricMapper;
    private final SubMetricMapper subMetricMapper;

    public DataInfoServiceImpl(DataInfoMapper dataInfoMapper, MetricMapper metricMapper, SubMetricMapper subMetricMapper) {
        this.dataInfoMapper = dataInfoMapper;
        this.metricMapper = metricMapper;
        this.subMetricMapper = subMetricMapper;
    }

    @Override
    public IPage<DataInfo> getDataInfoByPage(int pageNum) {
        Page<DataInfo> page = new Page<>(pageNum, 15);
        QueryWrapper<DataInfo> queryWrapper = new QueryWrapper<>();
        //queryWrapper.eq("is_deleted",0);
        return this.page(page, queryWrapper);
    }
    @Override
    public IPage<DataInfo> getDataInfoByQuestionTypeByPage(int pageNum, String questionType) {
        Page<DataInfo> page = new Page<>(pageNum, 15);
        QueryWrapper<DataInfo> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("question_type", questionType);
        return this.page(page, queryWrapper);
    }
    @Override
    public IPage<DataInfo> getDataInfoByPartContentByPage(int pageNum, String dimension, String metric) {
        Long metricId = metricMapper.selectIdByName(metric);
        Page<DataInfo> page = new Page<>(pageNum, 15);
        QueryWrapper<DataInfo> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("dimension", dimension)
                    .eq("metric_id", metricId);
        return this.page(page, queryWrapper);
    }

    @Override
    public IPage<DataInfo> getDataInfoByAllContentByPage(int pageNum, String dimension, String metric, String subMetric) {
        Long metricId = metricMapper.selectIdByName(metric);
        Long subMetricId = subMetricMapper.selectIdByName(subMetric);
        Page<DataInfo> page = new Page<>(pageNum, 15);
        QueryWrapper<DataInfo> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("dimension", dimension)
                    .eq("metric_id", metricId)
                    .eq("sub_metric_id", subMetricId);
        return this.page(page, queryWrapper);
    }

    @Override
    public DataInfo getDataInfoByDataId(Long dataId) {
        return dataInfoMapper.selectById(dataId);
    }
    @Override
    public boolean updateDataInfo(DataInfo dataInfo) {
        return dataInfoMapper.updateById(dataInfo) > 0;
    }

    @Override
    public boolean deleteDataInfoByDataId(Long dataId) {
        return dataInfoMapper.deleteById(dataId) > 0;
    }
}
