# 目录

<!-- TOC -->

- [目录](#目录)
- [Segmenter描述](#Segmenter描述)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [训练和测试](#训练和测试)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [ImageNet-1k上的Segmenter训练](#ImageNet-1k上的Segmenter训练)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# [Segmenter描述](#目录)

ViT采用纯Transformer架构，将图像分成多个patches进行输入，在很多图像分类任务中表现都不输最先进的卷积网络。
Segmenter作为一个纯Transformer的编码-解码架构，利用了模型每一层的全局图像上下文。
基于最新的ViT研究成果，将图像分割成块（patches），并将它们映射为一个线性嵌入序列，用编码器进行编码。再由Mask Transformer将编码器和类嵌入的输出进行解码，上采样后应用Argmax给每个像素一一分好类，输出最终的像素分割图。


# [数据集](#目录)

Cityscapes数据集包含5000幅高质量像素级别精细注释的街城市道场景图像。图像按2975/500/1525的分割方式分为三组，分别用于训练、验证和测试。数据集中共包含30类实体，其中19类用于验证。
- 下载数据集，目录结构如下：

 ```text
    ├─ cityscapes
    │  ├─ leftImg8bit
    │  │  ├─ train
    │  │  │  └─ [city_folders]
    │  │  └─ val
    │  │     └─ [city_folders]
    │  ├─ gtFine
    │  │  ├─ train
    │  │  │  └─ [city_folders]
    │  │  └─ val
    │  │     └─ [city_folders]
 ```

# [特性](#目录)

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.9/others/mixed_precision.html?highlight=%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6)
的训练方法，使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

# [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# [脚本说明](#目录)        │   

## 脚本及样例代码

```text
├── Segmenter
│   ├── eval.py
│   ├── npz2ckpt.ipynb
│   ├── README.md
│   ├── src
│   │   ├── args.py
│   │   ├── configs
│   │   │   ├── parser.py
│   │   │   └── segmenter.yaml
│   │   ├── data
│   │   │   ├── cityscapes_ms.py
│   │   │   ├── __init__.py
│   │   │   └── pipeline
│   │   │       ├── file_client.py
│   │   │       ├── formatting.py
│   │   │       ├── geometric.py
│   │   │       ├── __init__.py
│   │   │       ├── loading.py
│   │   │       └── transforms.py
│   │   ├── engine
│   │   │   ├── eval_utils.py
│   │   │   ├── __init__.py
│   │   │   └── train_engine.py
│   │   ├── models
│   │   │   ├── decoder.py
│   │   │   ├── encoder.py
│   │   │   ├── __init__.py
│   │   │   ├── layers
│   │   │   │   ├── attention.py
│   │   │   │   ├── drop_path.py
│   │   │   │   ├── ffn.py
│   │   │   │   ├── identity.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── weight_init.py
│   │   │   └── segmenter.py
│   │   └── tools
│   │       ├── amp.py
│   │       ├── criterion.py
│   │       ├── eval_callback.py
│   │       ├── get_misc.py
│   │       ├── helpers.py
│   │       ├── miou_metric.py
│   │       ├── moxing_adapter.py
│   │       ├── optim.py
│   │       ├── prepare_misc.py
│   │       └── schedulers.py
│   └── train.py
```

## 脚本参数

在segmenter.yaml中可以同时配置训练参数和评估参数。

- 配置Segmenter和ImageNet-1k数据集。

  ```text
    # ===== Architecture ===== #
    encoder: vit_large_patch16_384
    decoder: mask_transformer

    # ===== Dataset ===== #
    data_url: ../data/Cityscapes
    set: CityscapesMS
    num_classes: 19
    train_image_size: 768
    infer_image_size: 1024

    # ===== Learning Rate Policy ======== #
    optimizer: momentum
    base_lr: 0.01
    warmup_lr: 0.00001
    min_lr: 0.00001
    lr_scheduler: poly_lr
    warmup_length: 0


    # ===== Network training config ===== #
    amp_level: O1
    clip_global_norm_value: 5.
    is_dynamic_loss_scale: True
    epochs: 216
    weight_decay: 0.
    momentum: 0.9
    train_batch_size: 1
    eval_batch_size: 1
    encoder_drop_path_rate: 0.1
    encoder_dropout: 0.0
    decoder_drop_path_rate: 0.0
    decoder_dropout: 0.1
    pretrained: s3://open-data/pretrained/vit_large_patch16_384.ckpt

    # ===== Eval dataset config ===== #
    window_size: 768
    window_stride: 512
    crop_size: 768
    ignore_label: 255

    # ===== Hardware setup ===== #
    num_parallel_workers: 16
    device_target: Ascend
  ```

更多配置细节请参考脚本`segmenter.yaml`。 通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

# [训练和测试](#目录)

- Ascend处理器环境运行

  ```bash
  # 使用python启动单卡训练
  python train.py --device_id 0 --device_target Ascend --config ./src/configs/segmenter.yaml \
  > train.log 2>&1 &
  
  # 使用脚本启动单卡训练
  bash ./scripts/run_standalone_train_ascend.sh [DEVICE_ID] [CONFIG_PATH]
  
  # 使用脚本启动多卡训练
  bash ./scripts/run_distribute_train_ascend.sh [RANK_TABLE_FILE] [CONFIG_PATH]
  
  # 使用python启动单卡运行评估示例
  python eval.py --device_id 0 --device_target Ascend --config ./src/configs/segmenter.yaml --pretrained [CHECKPOINT_PATH]> ./eval.log 2>&1 &
  
  # 使用脚本启动单卡运行评估示例
  bash ./scripts/run_eval_ascend.sh [DEVICE_ID] [CONFIG_PATH] [CHECKPOINT_PATH]
  ```
  

对于分布式训练，需要提前创建JSON格式的hccl配置文件。

请遵循以下链接中的说明：

[hccl工具](https://gitee.com/mindspore/models/tree/master/utils/hccl_tools)


# [模型描述](#目录)

## 性能

### 评估性能

#### ImageNet-1k上的Segmenter训练

| 参数                 | Ascend                          |
| -------------------------- |---------------------------------|
|模型| Segmenter                            |
| 模型版本              | Segmenter                  |
| 资源                   | Ascend 910 8卡                   |
| 上传日期              | 2022-11-04                      |
| MindSpore版本          | 1.6.1                           |
| 数据集                    | Cityscape |
| 训练参数        | epoch=30, batch_size=8      |
| 优化器                  | AdamWeightDecay                 |
| 输出                    | MIOU                              |
| 分类准确率             | 八卡：0.7918      |
| 速度                      | 8卡：678.409 毫秒/步                  |
| 训练耗时          | 大约22小时（run on OpenI）       |


# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)