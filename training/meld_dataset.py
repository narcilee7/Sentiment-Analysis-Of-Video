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
    """MELD数据集加载器
    处理多模态情感对话数据集，包括视频、音频和文本
    """
    def __init__(self, csv_path, video_dir):
        """初始化数据集
        Args:
            csv_path: 标注文件路径，包含对话文本和标签
            video_dir: 视频文件目录
        """
        # 读取CSV文件中的标注数据
        self.data = pd.read_csv(csv_path)
        # 存储视频文件目录路径
        self.video_dir = video_dir
        # 初始化BERT分词器
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # 情感标签映射字典
        self.emotion_map = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surprise": 6
        }

        # 情感极性标签映射字典
        self.sentiment_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }

    def _load_video_frames(self, video_path):
        """加载视频帧
        从视频文件中提取帧，并进行预处理
        Args:
            video_path: 视频文件路径
        Returns:
            处理后的视频帧张量 [30, 3, 224, 224]
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            # 检查视频是否成功打开
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")
            
            # 读取第一帧检查视频完整性
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")
            
            # 重置视频读取位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # 读取30帧视频
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 调整帧大小为224x224并归一化
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)
            
        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")

        finally:
            cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames could be extracted from video: {video_path}")
        
        # 处理帧数不足或超过的情况
        if len(frames) < 30:
            # 帧数不足时用零帧填充
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            # 帧数过多时截断
            frames = frames[:30]

        # 转换维度顺序：[帧数, 高度, 宽度, 通道] -> [帧数, 通道, 高度, 宽度]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

    def _extract_audio_features(self, video_path):
        """提取音频特征
        从视频中提取音频并计算梅尔频谱图
        Args:
            video_path: 视频文件路径
        Returns:
            音频特征张量
        """
        # 构建音频文件路径
        audio_path = video_path.replace(".mp4", ".wav")
        
        try:
            # 使用ffmpeg从视频中提取音频
            ffmpeg_path = r"D:\code-utils\FFMpeg\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
            
            # 调用ffmpeg提取音频
            subprocess.run([
                ffmpeg_path,
                '-i', video_path,
                '-vn',  # 不处理视频
                '-acodec', 'pcm_s16le',  # 音频编码格式
                '-ar', '16000',  # 采样率16kHz
                '-ac', '1',  # 单声道
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 加载音频文件
            waveform, sample_rate = torchaudio.load(audio_path)

            # 重采样到16kHz（如果需要）
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # 计算梅尔频谱图
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,  # 梅尔滤波器数量
                n_fft=1024,  # FFT窗口大小
                hop_length=512  # 帧移
            )
            
            mel_spec = mel_spectrogram(waveform)
            
            # 标准化处理
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            # 统一特征长度为300
            if mel_spec.size(2) < 300:
                # 长度不足时进行填充
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                # 长度过长时截断
                mel_spec = mel_spec[:, :, :300]
        
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")
        
        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")
        
        finally:
            # 清理临时音频文件
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
        return mel_spec

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本
        Args:
            idx: 样本索引
        Returns:
            包含文本、视频、音频特征和标签的字典
        """
        # 处理tensor类型的索引
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        try:
            # 获取数据行
            row = self.data.iloc[idx]
            # 构建视频文件名
            video_filename = f"""dia{row["Dialogue_ID"]}_utt{row["Utterance_ID"]}.mp4"""
            path = os.path.join(self.video_dir, video_filename)
        
            # 检查视频文件是否存在
            if not os.path.exists(path):
                raise FileNotFoundError(f"No video file for filename: {path}")
            
            # 处理文本数据
            text_inputs = self.tokenizer(
                row['Utterance'],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # 加载视频帧
            video_frames = self._load_video_frames(path)
            # 提取音频特征
            audio_features = self._extract_audio_features(path)
            # 获取标签
            emotion_label = self.emotion_map[row["Emotion"].lower()]
            sentiment_label = self.sentiment_map[row["Sentiment"].lower()]

            # 返回处理后的样本数据
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
            return None

# 自定义collate函数，用于处理DataLoader中的批处理
def collate_fn(batch):
    """自定义批处理函数
    过滤掉无效的样本，并将有效样本组合成批次
    """
    # 过滤None值
    batch = list(filter(None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# 准备数据加载器
def prepare_dataloaders(train_csv, train_video_dir, dev_csv, dev_video_dir, test_csv, test_video_dir, batch_size=32):
    """准备数据加载器
    为训练、验证和测试集创建DataLoader
    """
    # 创建数据集对象
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    return train_loader, dev_loader, test_loader

# 测试代码
if __name__ == '__main__':
    # 准备数据加载器
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        "../dataset/train/train_sent_emo.csv",
        "../dataset/train/train_splits",
        "../dataset/dev/dev_sent_emo.csv",
        "../dataset/dev/dev_splits_complete",
        "../dataset/test/test_sent_emo.csv",
        "../dataset/test/output_repeated_splits_test",
    )
    
    # 测试第一个批次的数据
    for batch in train_loader:
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break