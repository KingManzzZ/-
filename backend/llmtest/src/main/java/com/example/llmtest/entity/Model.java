package com.example.llmtest.entity;


import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;

@Data
@TableName("model")
public class Model {

    @TableId(type = IdType.AUTO)
    @TableField("model_id")
    private Long modelId;

    @TableField("model_name")
    private String modelName;

    @TableField("model_description")
    private String modelDescription;
}
