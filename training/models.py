import torch.nn as nn
from meld_dataset import MELDDataset
from transformers import BertModel
from torchvision import models as vision_models
import torch
import os


class TextEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')

    for param in self.bert.parameters():
      # 冻结Bert的参数，只训练自己的参数
      param.requires_grad = False

    self.projection = nn.Linear(768, 128)
  
  def forward(self, input_ids, attention_mask):
    # Extract BERT embeddings
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

    # Use [CLS] token representation
    pooled_output = outputs.pooler_output

    return self.projection(pooled_output)
  

class VideoEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    # 使用预训练的R3D-18模型
    self.backbone = vision_models.video.r3d_18(pretrained=True)

    for param in self.backbone.parameters():
      # 冻结R3D-18模型的参数，只训练自己的参数
      param.requires_grad = False

    num_features = self.backbone.fc.in_features
    self.backbone.fc = nn.Sequential(
      nn.Linear(num_features, 128),
      nn.ReLU(),
      nn.Dropout(0.2)
    )
  
  def forward(self, x):
    # x: [batch_size, frames, channels, height, width] -> [batch_size, channels, frames, height, width]
    # 输入视频帧，输出特征向量 
    x = x.transpose(1, 2)
    return self.backbone(x)

class AudioEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    # 使用预训练的ResNet-18模型
    self.conv_layers = nn.Sequential(
      # Lower Level feature extraction
      nn.Conv1d(64, 64, kernel_size=3),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=2),
      # High Level feature extraction
      nn.Conv1d(64, 128, kernel_size=3),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.AdaptiveAvgPool1d(1)
    )

    for param in self.conv_layers.parameters():
      # 冻结ResNet-18模型的参数，只训练自己的参数
      param.requires_grad = False


    self.projection = nn.Sequential(
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Dropout(0.2)
    )

  def forward(self, x):
    x = x.squeeze(1)

    features = self.conv_layers(x)
    # Features: [batch_size, 128, 1]
    return self.projection(features.squeeze(-1))

# if __name__ == '__main__':
#   batch_size = 2
#   x = torch.randn(batch_size, 1, 64, 300)
#   print(f"x: {x.shape}")

#   x_squeezed = x.squeeze(1)

#   print(f"x_squeezed: {x_squeezed.shape}")

class MultimodelSentimentModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.text_encoder = TextEncoder()

    self.video_encoder = VideoEncoder()

    self.audio_encoder = AudioEncoder()

    # Fusion Layer
    self.fusion_layer = nn.Sequential(
      nn.Linear(128 * 3, 256),
      nn.BatchNorm1d(256),
      nn.ReLU(),
      nn.Dropout(0.3)
    )

    # classification layers
    self.emotion_classifier = nn.Sequential(
      nn.Linear(256, 64),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(64, 7),
    )

    self.sentiment_classifier = nn.Sequential(
      nn.Linear(256, 64),
      nn.ReLU(),
      nn.Dropout(0.2),
      nn.Linear(64, 3),
    )
  
  def forward(self, text_inputs, video_frames, audio_features):
    text_features = self.text_encoder(text_inputs['input_ids'], text_inputs['attention_mask'])

    video_features = self.video_encoder(video_frames)

    audio_features = self.audio_encoder(audio_features)

    combined_features = torch.cat((text_features, video_features, audio_features), dim=-1) # [batch_size, 128 * 3]

    fused_features = self.fusion_layer(combined_features)

    emotion_output = self.emotion_classifier(fused_features)

    sentiment_output = self.sentiment_classifier(fused_features)

    return {
      "emotions": emotion_output,
      "sentiments": sentiment_output
    }

if __name__ == "__main__":
  dataset = MELDDataset("../dataset/train/train_sent_emo.csv", "../dataset/train/train_splits")

  sample = dataset[0]

  model = MultimodelSentimentModel()

  model.eval()
  
  text_inputs = {
    "input_ids": sample['text_inputs']['input_ids'].unsqueeze(0),
    "attention_mask": sample['text_inputs']['attention_mask'].unsqueeze(0),
  }

  video_frames = sample['video_frames'].unsqueeze(0)
  audio_features = sample['audio_features'].unsqueeze(0)

  with torch.inference_mode():
    outputs = model(text_inputs, video_frames, audio_features)

    emotion_probs = torch.softmax(outputs['emotions'], dim=1)[0]
    sentiment_probs = torch.softmax(outputs['sentiments'], dim=1)[0]

  # 使用与数据集类相同的映射
  emotion_map = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6
  }

  sentiment_map = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
  }

  # 反转映射，从数字到文本
  emotion_map_reverse = {v: k for k, v in emotion_map.items()}
  sentiment_map_reverse = {v: k for k, v in sentiment_map.items()}

  print("\nEmotion Probabilities:")
  for i, prob in enumerate(emotion_probs):
    print(f"{emotion_map_reverse[i]}: {prob:.2f}")

  print("\nSentiment Probabilities:")
  for i, prob in enumerate(sentiment_probs):
    print(f"{sentiment_map_reverse[i]}: {prob:.2f}")

  print("\nPrediction for utterance")

  def __getitem__(self, idx):
    # 如果idx是tensor，转换为python标量
    if isinstance(idx, torch.Tensor):
      idx = idx.item()
    try:
      # 获取数据行
      row = self.data.iloc[idx]
      # 构建视频文件名
      video_filename = f"""dia{row["Dialogue_ID"]}_utt{row["Utterance_ID"]}.mp4"""

      path = os.path.join(self.video_dir, video_filename)
    
      if not os.path.exists(path):
        raise FileNotFoundError(f"No video file for filename: {path}")
      
      # 对文本进行tokenize
      text_inputs = self.tokenizer(row['Utterance'],
                                   padding="max_length",
                                   truncation=True,
                                   max_length=128,
                                   return_tensors="pt")
      
      # 加载视频帧
      video_frames = self._load_video_frames(path)

      # 提取音频特征
      audio_features = self._extract_audio_features(path)

      # 转换情感和情感极性标签
      emotion_label = self.emotion_map[row["Emotion"].lower()]
      sentiment_label = self.sentiment_map[row["Sentiment"].lower()]

      # 返回样本数据
      return {
        'text_inputs': {
          'input_ids': text_inputs['input_ids'].squeeze(),
          'attention_mask': text_inputs['attention_mask'].squeeze(),
        },
        'video_frames': video_frames,
        'audio_features': audio_features,
        'emotion_label': torch.tensor(emotion_label),
        'sentiment_label': torch.tensor(sentiment_label),
      }
    except Exception as e:
      print(f"Error processing {path}: {str(e)}")
      return None  # 修改这里：返回None而不是raise None