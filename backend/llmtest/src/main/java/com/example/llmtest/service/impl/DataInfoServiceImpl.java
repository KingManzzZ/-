package com.example.llmtest.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.example.llmtest.exceptionhandler.BusinessException;
import com.example.llmtest.exceptionhandler.ReturnCode;
import com.example.llmtest.pojo.dto.DataInfoPageQueryDTO;
import com.example.llmtest.pojo.entity.DataInfo;
import com.example.llmtest.mapper.DataInfoMapper;
import com.example.llmtest.mapper.MetricMapper;
import com.example.llmtest.mapper.SubMetricMapper;
import com.example.llmtest.pojo.enums.QuestionTypeEnum;
import com.example.llmtest.pojo.vo.DataInfoVO;
import com.example.llmtest.service.DataInfoService;
import com.example.llmtest.utils.CustomUtil;
import org.apache.commons.lang3.StringUtils;
import org.springframework.stereotype.Service;

import java.util.Map;



@Service
public class DataInfoServiceImpl extends ServiceImpl<DataInfoMapper, DataInfo> implements DataInfoService {

    private final DataInfoMapper dataInfoMapper;
    private final MetricMapper metricMapper;
    private final SubMetricMapper subMetricMapper;
    private final CustomUtil customUtil;

    public DataInfoServiceImpl(DataInfoMapper dataInfoMapper, MetricMapper metricMapper,
                               SubMetricMapper subMetricMapper, CustomUtil customUtil) {
        this.dataInfoMapper = dataInfoMapper;
        this.metricMapper = metricMapper;
        this.subMetricMapper = subMetricMapper;
        this.customUtil = customUtil;
    }

    /**
     * 根据条件分页查询题目
     * @param queryDTO
     * @return
     */
    @Override
    public IPage<DataInfoVO> getDataInfoByConditions(DataInfoPageQueryDTO queryDTO) {

        if (StringUtils.isNotBlank(queryDTO.getSubMetric())) {
            if (StringUtils.isBlank(queryDTO.getMetric())) {
                throw new BusinessException(ReturnCode.RC400.getCode(), "指定子指标时，指标不能为空");
            }
        }
        if (StringUtils.isNotBlank(queryDTO.getQuestionType())) {
            if (!QuestionTypeEnum.contains(queryDTO.getQuestionType())) {
                throw new BusinessException(ReturnCode.RC400.getCode(), "题型不存在");
            }
        }
        Map<String, String> dimensionMap = customUtil.getDimensionMap();
        Map<String, String> metricMap = customUtil.getMetricMap();
        Map<String, String> subMetricMap = customUtil.getSubMetricMap();
        if (StringUtils.isNotBlank(queryDTO.getDimension())) {
            if (!dimensionMap.containsValue(queryDTO.getDimension())) {
                throw new BusinessException(ReturnCode.RC400.getCode(), "维度不存在");
            }
        }
        if (StringUtils.isNotBlank(queryDTO.getMetric())) {
            if (!metricMap.containsValue(queryDTO.getMetric())) {
                throw new BusinessException(ReturnCode.RC400.getCode(), "指标不存在");
            }
        }
        if (StringUtils.isNotBlank(queryDTO.getSubMetric())) {
            if (!subMetricMap.containsValue(queryDTO.getSubMetric())) {
                throw new BusinessException(ReturnCode.RC400.getCode(), "子指标不存在");
            }
        }

        // 正常查询

        long pageNum = queryDTO.getPageNum() == null ? 1 : queryDTO.getPageNum();
        long pageSize = 15;

        Page<DataInfo> page = new Page<>(pageNum, pageSize);
        QueryWrapper<DataInfo> queryWrapper = new QueryWrapper<>();
        if (StringUtils.isNotBlank(queryDTO.getQuestionType())) {
            queryWrapper.eq("question_type", queryDTO.getQuestionType());
        }
        if (StringUtils.isNotBlank(queryDTO.getDimension())) {
            queryWrapper.eq("dimension", queryDTO.getDimension());
        }
        if (StringUtils.isNotBlank(queryDTO.getMetric())) {
            Long metricId = metricMapper.selectIdByName(queryDTO.getMetric());
            if (metricId != null) {
                queryWrapper.eq("metric_id", metricId);
            }
        }
        if (StringUtils.isNotBlank(queryDTO.getSubMetric())) {
            Long subMetricId = subMetricMapper.selectIdByName(queryDTO.getSubMetric());
            if (subMetricId != null) {
                queryWrapper.eq("sub_metric_id", subMetricId);
            }
        }
        IPage<DataInfo> dataInfoPage = this.page(page, queryWrapper);
        IPage<DataInfoVO> voPage = dataInfoPage.convert(customUtil::convertToDataInfoVO);
        return voPage;
    }

    /**
     * 根据da题目Id查询题目
     * @param dataId
     * @return
     */
    @Override
    public DataInfoVO getDataInfoByDataId(Long dataId) {
        return customUtil.convertToDataInfoVO(dataInfoMapper.selectById(dataId));
    }


    @Override
    public boolean updateDataInfo(DataInfo dataInfo) {
        return dataInfoMapper.updateById(dataInfo) > 0;
    }

}
