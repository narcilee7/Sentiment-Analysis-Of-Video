import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
import torch
import os
from sklearn.metrics import precision_score ,accuracy_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from .meld_dataset import MELDDataset

class TextEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    # 使用预训练的BERT模型
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    # 冻结BERT参数，只训练投影层
    for param in self.bert.parameters():
      param.requires_grad = False
    # 将768维BERT输出投影到128维
    self.projection = nn.Linear(768, 128)
  
  def forward(self, input_ids, attention_mask):
    # 获取BERT的输出
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    # 使用[CLS]标记的输出作为整个序列的表示
    pooled_output = outputs.pooler_output
    # 投影到低维空间
    return self.projection(pooled_output)
  

class VideoEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    # 使用预训练的R3D-18模型（3D ResNet）
    self.backbone = vision_models.video.r3d_18(pretrained=True)
    # 冻结预训练模型参数
    for param in self.backbone.parameters():
      param.requires_grad = False
    # 替换最后的全连接层
    num_features = self.backbone.fc.in_features
    self.backbone.fc = nn.Sequential(
      nn.Linear(num_features, 128),
      nn.ReLU(),
      nn.Dropout(0.2)
    )
  
  def forward(self, x):
    # 调整维度顺序以适应R3D-18的输入要求
    x = x.transpose(1, 2)
    return self.backbone(x)

class AudioEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    # 音频特征提取网络
    self.conv_layers = nn.Sequential(
      # 低级特征提取
      nn.Conv1d(64, 64, kernel_size=3),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=2),
      # 高级特征提取
      nn.Conv1d(64, 128, kernel_size=3),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.AdaptiveAvgPool1d(1)
    )
    # 投影层
    self.projection = nn.Sequential(
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Dropout(0.2)
    )
  
  def forward(self, x):
    x = x.squeeze(1)
    features = self.conv_layers(x)
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

class MultiModalTrainer:
  def __init__(self, model, train_loader, val_loader):
    self.model = model 
    self.train_loader = train_loader
    self.val_loader = val_loader

    # Log dataset sized
    tran_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    print("\nDataset sizes:")
    print(f"Training: {tran_size}")
    print(f"Validation: {val_size}\n")
    print(f"Batches per epoch: {len(train_loader)},")

    timestamp = datetime.now().strftime("%b%d_%H-%M-%S") # Dec17_09-09-09
    base_dir = '/opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'
    log_dir = f"{base_dir}/run_{timestamp}"
    self.writer = SummaryWriter(log_dir=log_dir)
    self.global_step = 0

    # Very hight: 1, high: 01-0.01, medium: 1e-1, low: 1e-4, very low: 1e-5
    self.optimizer = torch.optim.Adam([
      {"params": model.text_encoder.parameters(), "lr": 8e-6},
      {"params": model.video_encoder.parameters(), "lr": 8e-5},
      {"params": model.audio_encoder.parameters(), "lr": 8e-5},
      {"params": model.fusion_layer.parameters(), "lr": 5e-4},
      {"params": model.emotion_classifier.parameters(), "lr": 5e-4},
      {"params": model.sentiment_classifier.parameters(), "lr": 5e-4},
    ], weight_decay=1e-5)


    self.sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      self.optimizer,
      mode='min', 
      patience=2,
      factor=0.1,
    )

    self.current_train_losses = None

    self.emotion_criterion = nn.CrossEntropyLoss(
      label_smoothing = 0.05
    )

    self.sentiment_criterion = nn.CrossEntropyLoss(
      label_smoothing = 0.05
    )

  def log_metrics(self, losses, metrics=None, phase="train"):
    if phase == "train":
      self.current_train_losses = losses
    else:
      # Validation phase
      self.writer.add_scalar(
        "loss/total/train",
        self.current_train_losses["total"],
        self.global_step
      )
      self.writer.add_scalar(
        "loss/total/val",
        losses["total"],
        self.global_step
      )
      self.writer.add_scalar(
        "loss/emotion/train",
        self.current_train_losses["emotion"],
        self.global_step
      )
      self.writer.add_scalar(
        "loss/emotion/val",
        losses["emotion"],
        self.global_step
      )
      self.writer.add_scalar(
        "loss/sentiment/train",
        self.current_train_losses["sentiment"],
        self.global_step
      )
      self.writer.add_scalar(
        "loss/sentiment/val",
        losses["sentiment"],
        self.global_step
      )
    if metrics:
      self.writer.add_scalar(
        f'{phase}/emotion_precision',
        metrics['emotion_precision'],
        self.global_step
      )
      self.writer.add_scalar(
        f'{phase}/emotion_accuracy',
        metrics['emotion_accuracy'],
        self.global_step
      )
      self.writer.add_scalar(
        f'{phase}/sentiment_precision',
        metrics['sentiment_precision'],
        self.global_step
      )
      self.writer.add_scalar(
        f'{phase}/sentiment_accuracy',
        metrics['sentiment_accuracy'],
        self.global_step
      )

  def train_epoch(self):
    self.model.train()
    running_loss = {
      "total": 0,
      "emotion": 0,
      "sentiment": 0
    }

    for batch in self.train_loader:
      device = next(self.model.parameters()).device
      text_inputs = {
        "input_ids": batch["text_input"]["input_ids"].to(device),
        "attention_mask": batch["text_input"]["attention_mask"].to(device)
      }
      video_frames = batch["video_frames"].to(device)
      audio_features = batch["audio_features"].to(device)
      emotion_labels = batch["emotion_labels"].to(device)
      sentiment_labels = batch["sentiment_labels"].to(device)
      # Zero gradients
      self.optimizer.zero_grad()

      # Forward pass
      outputs = self.model(text_inputs, video_frames, audio_features)

      # Calculate losses using raw logits
      emotion_loss = self.emotion_criterion(outputs["emotions"], emotion_labels)

      sentiment_loss = self.sentiment_criterion(outputs["sentiment"], sentiment_labels)

      total_loss = emotion_loss + sentiment_loss

      # Backward pass, Calculate gradients
      total_loss.backward()

      # Gradient clipping
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

      self.optimizer.step()

      # Track losses
      running_loss["total"] += total_loss.item()
      running_loss["emotion"] += emotion_loss.item()
      running_loss["sentiment"] += sentiment_loss.item()

      self.log_metrics({
        "total": total_loss.item(),
        "emotion": emotion_loss.item(),
        "sentiment": sentiment_loss.item()
      })

      self.global_step += 1

    return {
      k: v/len(self.train_loader) for k, v in running_loss.items()
    }
  
  def evaluate(self, data_loader, phase="val"):
    self.model.eval()
    losses = {
      "total": 0,
      "emotion": 0,
      "sentiment": 0
    }
    all_emotion_preds = []
    all_emotion_labels = []
    all_sentiment_preds = []
    all_sentiment_labels = []

    with torch.inference_mode():
      for batch in data_loader:  
        device = next(self.model.parameters()).device
        text_inputs = {
          "input_ids": batch["text_input_ids"].to(device),
          "attention_mask": batch["text_attention_mask"].to(device)
        }
        video_frames = batch["video_frames"].to(device)
        audio_features = batch["audio_features"].to(device)
        emotion_labels = batch["emotion_labels"].to(device)
        sentiment_labels = batch["sentiment_labels"].to(device)

        outputs = self.model(text_inputs, video_frames, audio_features)

        emotion_loss = self.emotion_criterion(
          outputs["emotions"], emotion_labels
        )
        sentiment_loss = self.sentiment_criterion(
          outputs["sentiments"], sentiment_labels
        )
        total_loss = emotion_loss + sentiment_loss

        # [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]
        predictions = [1, 0]

        all_emotion_preds.extend(
          outputs["emotions"].argmax(dim=-1).cpu().numpy()
        )

        all_emotion_labels.extend(emotion_labels.cpu().numpy())

        all_sentiment_preds.extend(
          outputs["sentiments"].argmax(dim=-1).cpu().numpy()
        )

        all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

        # Track losses
        losses['total'] += total_loss.item()
        losses['emotion'] += emotion_loss.item()
        losses['sentiment'] += sentiment_loss.item()

    avg_loss = {k: v/ len(data_loader) for k, v in losses.items()}

    # Compute the precision and accuracy
    emotion_precision = precision_score(
      all_emotion_labels,
      all_emotion_preds,
      average='weighted'
    )
    emotion_accuracy = accuracy_score(
      all_emotion_labels,
      all_emotion_preds
    )
    sentiment_precision = precision_score(
      all_sentiment_labels,
      all_sentiment_preds,
      average='weighted'
    )
    sentiment_accuracy = accuracy_score(
      all_sentiment_labels,
      all_sentiment_preds
    )

    self.log_metrics(avg_loss, {
      'emotion_precision': emotion_precision,
      'emotion_accuracy': emotion_accuracy,
      'sentiment_precision': sentiment_precision,
      'sentiment_accuracy': sentiment_accuracy
    }, phase=phase)

    if phase == 'val':
      avg_loss = {k: v.avg for k, v in self.val_loss.items()}

    return avg_loss, {
      'emotion_precision': emotion_precision,
      'emotion_accuracy': emotion_accuracy,
      'sentiment_precision': sentiment_precision,
      'sentiment_accuracy': sentiment_accuracy
    }

