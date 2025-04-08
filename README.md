# Sentiment analysis of video

## Dataset

Melt-Sentiment-Analysis-Dataset

视频、音频、文本

视频通过 ResNet50 提取特征，音频通过Audio encoder Raw spectrogam、文本通过Text encoder BERT

Fused 将视频、音频、文本特征进行融合。

采用late Fusion 进行融合。