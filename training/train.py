import os
import argparse
import torchaudio
import torch
import tqdm
import json
import sys

from meld_dataset import prepare_dataloaders
from models import MultimodelSentimentModel, MultiModalTrainer
from install_ffmpeg import install_ffmpeg

# AWS SageMaker环境变量配置
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', ".")  # 模型保存目录
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', "/opt/ml/input/data/training")  # 训练数据目录
SM_CHANNEL_VALIDATION = os.environ.get('SM_CHANNEL_VALIDATION', "/opt/ml/input/data/validation")  # 验证数据目录
SM_CHANNEL_TEST = os.environ.get('SM_CHANNEL_TEST', "/opt/ml/input/data/test")  # 测试数据目录

# 配置PyTorch CUDA内存分配策略
os.environ['PYTORCH_CUDE_ALLOC_CONF'] = "expandable_segments:True"

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)

    # 数据目录参数
    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)

    return parser.parse_args()

def main():
    """主训练流程"""
    # 安装并验证FFmpeg
    if not install_ffmpeg():
        print("Error: FFmpeg installation failed. Cannot continue training.")
        sys.exit(1)

    # 检查可用的音频后端
    print("Available audio backends: ")
    print(str(torchaudio.list_audio_backends()))

    # 解析命令行参数
    args = parse_args()

    # 设置设备（GPU/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # 如果使用GPU，跟踪内存使用情况
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Initial GPU memory used: {memory_used:.2f} GB")

    # 准备数据加载器
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir=os.path.join(args.train_dir, 'train_splits'),
        dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),
        dev_video_dir=os.path.join(args.test_dir, "dev_splits_complete"),
        test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),
        test_video_dir=os.path.join(args.test_dir, 'output_repeated_splits_test'),
        batch_size=args.batch_size,
    )
    
    # 打印数据路径信息
    print(f"Training CSV path: {os.path.join(args.train_dir, 'train_sent_emo.csv')}")
    print(f"Training video directory: {os.path.join(args.train_dir, 'train_splits')}")

    # 初始化模型并移至指定设备
    model = MultimodelSentimentModel().to(device)

    # 创建训练器
    trainer = MultiModalTrainer(model, train_loader, val_loader)

    # 记录最佳验证损失
    best_val_loss = float("inf")

    # 用于记录训练过程中的指标
    metrics_data = {
        "train_losses": [],
        "val_losses": [],
        "epochs": [],
    }

    # 训练循环
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        # 训练一个epoch
        train_loss = trainer.train_epoch()
        # 在验证集上评估
        val_loss, val_metrics = trainer.evaluate(val_loader)

        # 记录训练指标
        metrics_data['train_losses'].append(train_loss['total'])
        metrics_data['val_losses'].append(val_loss['total'])
        metrics_data['epochs'].append(epoch)

        # 以SageMaker格式记录指标
        print(json.dumps({
            "metrics": [
                {"Name": "train:loss", "Value": train_loss["total"]},
                {"Name": "validation:loss", "Value": val_loss["total"]},
                {"Name": "validation:emotion_precision", "Value": val_metrics["emotion_precision"]},
                {"Name": "validation:emotion_accuracy", "Value": val_metrics["emotion_accuracy"]},
                {"Name": "validation:sentiment_precision", "Value": val_metrics["sentiment_precision"]},
                {"Name": "validation:sentiment_accuracy", "Value": val_metrics["sentiment_accuracy"]},
            ]
        }))

        # 记录GPU内存使用情况
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory used: {memory_used:.2f} GB")

        # 保存最佳模型
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
    
    # 在测试集上进行最终评估
    print("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    metrics_data["test_loss"] = test_loss["total"]
    
    # 记录测试集结果
    print(json.dumps({
        "metrics": [
            {"Name": "test:loss", "Value": test_loss["total"]},
            {"Name": "test:emotion_accuracy", "Value": test_loss["emotion_accuracy"]},
            {"Name": "test:sentiment_accuracy", "Value": test_loss["sentiment_accuracy"]},
            {"Name": "test:emotion_precision", "Value": test_loss["emotion_precision"]},
            {"Name": "test:sentiment_precision", "Value": test_loss["sentiment_precision"]},
        ]
    }))

if __name__ == "__main__":
    main()