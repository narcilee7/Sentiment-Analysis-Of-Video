from models import MultimodelSentimentModel

def count_parameters(model):
    """统计模型各个组件的参数数量
    
    Args:
        model: 多模态情感分析模型实例
    
    Returns:
        tuple: (参数统计字典, 总参数数量)
        - 参数统计字典包含每个组件的参数数量
        - 总参数数量为所有参数的总和
    """
    # 初始化各组件参数计数器
    params_dict = {
        'text_encoder': 0,
        'video_encoder': 0,
        'audio_encoder': 0,
        'fusion_layer': 0,
        'emotion_classifier': 0,
        'sentiment_classifier': 0
    }

    total_params = 0

    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        # 计算当前参数的数量
        param_count = param.numel()
        total_params += param.numel()
        
        # 根据参数名称将参数数量加到对应的组件计数器中
        if 'text_encoder' in name:
            params_dict['text_encoder'] += param_count
        elif 'video_encoder' in name:
            params_dict['video_encoder'] += param_count
        elif 'audio_encoder' in name:
            params_dict['audio_encoder'] += param_count
        elif 'fusion_layer' in name:
            params_dict['fusion_layer'] += param_count
        elif 'emotion_classifier' in name:
            params_dict['emotion_classifier'] += param_count
        elif 'sentiment_classifier' in name:
            params_dict['sentiment_classifier'] += param_count

    return params_dict, total_params


if __name__ == '__main__':
    # 创建模型实例
    model = MultimodelSentimentModel()

    # 统计参数
    param_dics, total_params = count_parameters(model)

    # 打印每个组件的参数数量
    print("Parameter Count by component")
    for component, count in param_dics.items():
        print(f"{component:20s}: {count:,} parameters")

    # 打印总参数数量
    print("\nTotal trainable parameters: ", f"{total_params:,}")