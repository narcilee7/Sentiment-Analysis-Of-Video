# Sentiment analysis of video

## Dataset

Melt-Sentiment-Analysis-Dataset

视频、音频、文本

视频通过 ResNet50 提取特征，音频通过Audio encoder Raw spectrogam、文本通过Text encoder BERT

Fused 将视频、音频、文本特征进行融合。

采用late Fusion 进行融合。

## 依赖包

- torch
- torchaudio
- torchvision
- transformers
- pandas
- opencv-python

## 训练策略

video encoder Resnet3D 18 layer, Text encoder BERT, Audio encoder Raw spectrogam -> Concatenate data -> Fusion Layer -> Emotion Classifier and Sentiment Classifier

### Trainable layers

1. Linear: Fully connected layer Classification
2. Conv1d: Lears patters in Sequential data Eg: audio

### Functional layers - Dosn't learn

1. RELU: Adds non-linearity
2. Dropout: Prevent overfitting by randomly "turning off" some neurons
3. MaxPool1d: Only keeps strongest features (Input:[1, 2, 3, 4] -> Output:[2, 4])
4. AdaptiveAvgPoo1d: Averges dimension Input:[1, 2, 3, 4] -> Output:[2.5]

### Normalization Layers

1. BatchNorm1d: Normalizes values Input:[1, 2, 3, 4] -> Output:[0.2, 0.5,- 0.3, 1.1]
