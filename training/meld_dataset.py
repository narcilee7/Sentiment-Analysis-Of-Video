from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio

# MELD数据集处理类
class MELDDataset(Dataset):
  def __init__(self, csv_path, video_dir):
    self.data = pd.read_csv(csv_path)
    self.video_dir = video_dir
    self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    self.emotion_map = {
      "anger": 0,
      "disgust": 1,
      "fear": 2,
      "joy": 3,
      "neutral": 4,
      "sadness": 5,
      "surprise": 6
    }

    self.sentiment_map = {
      'negative': 0,
      'neutral': 1,
      'positive': 2
    }

  #  加载视频帧
  def _load_video_frames(self, video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    try:
      if not cap.isOpened():
        raise ValueError(f"Video not found: {video_path}")
      
      # 读取视频帧
      ret, frame = cap.read()

      if not ret or frame is None:
        raise ValueError(f"Video not found: {video_path}")
      
      # 重置索引，保证不跳过第一帧
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      
      while len(frames) < 30 and cap.isOpened():
        ret, frame = cap.read()

        if not ret:
          break
        
        # 224 224 高度
        frame = cv2.resize(frame, (224, 224))

        # RGB化
        frame = frame / 255.0  # 归一化到 [0, 1] 范围

        frames.append(frame)
      
    except Exception as e:
      raise ValueError(f"Video error: {str(e)}")

    finally:
      cap.release()
    
    if (len(frames) == 0):
      raise ValueError(f"No frames could be extracted from video: {video_path}")
    
    if len(frames) < 30:
      # 填充虚拟帧
      frames += [np.zeros_like(frames[0])] * (30 - len(frames))
    else:
      # 截断一下
      frames = frames[:30]

    # Before permute: [frames, height, width, channels]
    # Afer permute: [frames, channels, height, width]
    return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

  def _extract_audio_features(self, video_path):
    # 音频路径
    audio_path = video_path.replace(".mp4", ".wav")
    
    try:
      # 使用完整路径
      ffmpeg_path = r"D:\code-utils\FFMpeg\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
      
      subprocess.run([
        ffmpeg_path,
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        audio_path
      ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

      waveform, sample_rate = torchaudio.load(audio_path)

      if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
      
      mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=64,
        n_fft=1024,
        hop_length=512
      )
      
      mel_spec = mel_spectrogram(waveform)
      
      # Normalize
      mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

      if mel_spec.size(2) < 300:
        padding = 300 - mel_spec.size(2)
        mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
      else:
        mel_spec = mel_spec[:, :, :300]
    
    except subprocess.CalledProcessError as e:
      raise ValueError(f"Audio extranction error: {str(e)}")
    
    except Exception as e:
      raise ValueError(f"Audio error: {str(e)}")
    
    finally:
      if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"Deleted audio file: {audio_path}")
      
    return mel_spec

  # override
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    if isinstance(idx, torch.Tensor):
      idx = idx.item()
    try:
      # 数据结构 Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime
      # 获取单行文本对话
      row = self.data.iloc[idx]
      # 根据row id 匹配视频文件名
      video_filename = f"""dia{row["Dialogue_ID"]}_utt{row["Utterance_ID"]}.mp4"""

      path = os.path.join(self.video_dir, video_filename)
    
      if not os.path.exists(path):
        raise FileNotFoundError(f"No video file for filename: {path}")
      
      # 进行标注
      text_inputs = self.tokenizer(row['Utterance'],
                                   padding="max_length",
                                   truncation=True,
                                   max_length=128,
                                   return_tensors="pt")
      
      video_frames = self._load_video_frames(path)

      # 提取音频特征
      audio_features = self._extract_audio_features(path)

      # Map sentiment and emotion labels
      emotion_label = self.emotion_map[row["Emotion"].lower()]
      sentiment_label = self.sentiment_map[row["Sentiment"].lower()]

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
      raise None

def collate_fn(batch):
  # Filter out None samples
  batch = list(filter(None, batch))
  return torch.utils.data.dataloader.default_collate(batch)
  pass

# 预加载数据集
def prepare_dataloaders(train_csv, train_video_dir, dev_csv, dev_video_dir, test_csv, test_video_dir, batch_size=32):
  train_dataset = MELDDataset(train_csv, train_video_dir)
  dev_dataset = MELDDataset(dev_csv, dev_video_dir)
  test_dataset = MELDDataset(test_csv, test_video_dir)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn) # shuffle代表是否打乱数据
  dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

  return train_loader, dev_loader, test_loader



if __name__ == '__main__':
  # 使用绝对路径
  # meld = MELDDataset("../dataset/dev/dev_sent_emo.csv",
  #                     "../dataset/dev/dev_splits_complete")

  train_loader, dev_loader, test_loader = prepare_dataloaders("../dataset/train/train_sent_emo.csv",
                                                              "../dataset/train/train_splits",
                                                              "../dataset/dev/dev_sent_emo.csv",
                                                              "../dataset/dev/dev_splits_complete",
                                                              "../dataset/test/test_sent_emo.csv",
                                                              "../dataset/test/output_repeated_splits_test",
  )
  for batch in train_loader:
    print(batch['text_inputs'])
    print(batch['video_frames'].shape)
    print(batch['audio_features'].shape)
    print(batch['emotion_label'])
    print(batch['sentiment_label'])
    break