# 视频情感分析模型

## 项目概述

这是一个多模态情感分析模型，可以同时分析视频中的视觉、音频和文本信息，从而进行情感和情感极性的分类。

## 数据集

使用 MELD (Multimodal EmotionLines Dataset) 数据集，包含：

- 视频数据
- 音频数据
- 文本对话数据

### 标签类别

- 情感分类 (7类)：
  - anger (愤怒)
  - disgust (厌恶)
  - fear (恐惧)
  - joy (快乐)
  - neutral (中性)
  - sadness (悲伤)
  - surprise (惊讶)

- 情感极性分类 (3类)：
  - negative (消极)
  - neutral (中性)
  - positive (积极)

## 模型架构

### 1. 特征提取器

- 视频特征：使用预训练的 ResNet3D-18 提取视觉特征
- 音频特征：使用自定义 CNN 网络从梅尔频谱图中提取特征
- 文本特征：使用预训练的 BERT 模型提取文本特征

### 2. 特征融合

采用 Late Fusion 策略：
1. 各模态特征提取器输出 128 维特征向量
2. 将三个特征向量拼接为 384 维向量
3. 通过融合层降维至 256 维

### 3. 分类器

- 情感分类器：256维 -> 64维 -> 7类输出
- 情感极性分类器：256维 -> 64维 -> 3类输出

## 依赖环境

主要依赖包：
- torch
- torchaudio
- torchvision
- transformers (BERT)
- pandas
- opencv-python
- ffmpeg-python

## 训练流程

1. 数据预处理：
   - 视频：提取30帧，调整为224x224大小
   - 音频：提取梅尔频谱图特征
   - 文本：使用BERT tokenizer处理

2. 训练参数：
   - 批次大小：16
   - 学习率：1e-4
   - 训练轮数：20

3. 训练策略：
   - 使用验证集监控训练过程
   - 保存验证损失最低的模型
   - 支持GPU训练加速
   - 使用TensorBoard记录训练指标

4. 评估指标：
   - 情感分类准确率和精确率
   - 情感极性分类准确率和精确率

## 文件结构

- `models.py`: 模型定义
- `meld_dataset.py`: 数据集加载和预处理
- `train.py`: 训练脚本
- `install_ffmpeg.py`: FFmpeg安装脚本
- `count_parameters.py`: 模型参数统计工具

## AWS SageMaker 支持

本项目支持在 AWS SageMaker 上训练，包含相关环境变量配置。

## 使用说明

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 准备数据：
   - 将MELD数据集放在指定目录
   - 确保视频文件和CSV标注文件对应

3. 开始训练：
   ```bash
   python training/train.py
   ```

4. 监控训练：
   - 通过控制台输出查看训练和验证指标
   - 使用TensorBoard查看详细训练过程

## 注意事项

1. 确保安装了FFmpeg
2. 需要足够的GPU内存（建议8GB以上）
3. 视频处理可能需要较长时间
4. 建议使用SSD存储数据集以加快数据加载速度
