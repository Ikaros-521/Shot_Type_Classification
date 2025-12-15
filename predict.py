#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
镜头类型分类预测脚本
使用训练好的PyTorch模型对图像进行镜头类型分类
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import sys
import time

# 类别映射
CLASS_MAPPING = {
    0: "远景 (Long shot, LS)",
    1: "全景 (Full shot, FS)", 
    2: "中景 (Medium shot, MS)",
    3: "特写 (Close-up shot, CS)",
    4: "极特写 (Extreme close-up shot, ECS)"
}

def load_model(model_path, device):
    """加载训练好的模型"""
    try:
        model = torch.load(model_path, map_location=device)
        model.eval()
        print(f"✓ 成功加载模型: {model_path}")
        return model
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        sys.exit(1)

def preprocess_image(image_path):
    """图像预处理"""
    try:
        # 图像预处理变换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 加载并转换图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        print(f"✗ 图像处理失败: {e}")
        return None

def predict(image_path, model, device):
    """进行预测"""
    # 预处理图像
    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        return None
    
    # 移动到设备
    image_tensor = image_tensor.to(device)
    
    # 预测
    with torch.no_grad():
        start_time = time.time()
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        print(f"✓ 预测完成，耗时: {time.time() - start_time:.4f} 秒")
        
        return {
            'class_id': predicted.item(),
            'class_name': CLASS_MAPPING[predicted.item()],
            'confidence': confidence.item()
        }

def main():
    parser = argparse.ArgumentParser(
        description='镜头类型分类预测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python predict.py image.jpg                    # 预测单张图像
  python predict.py image.jpg --verbose          # 显示详细信息
  python predict.py image.jpg --model custom.pt  # 使用自定义模型
        """
    )
    
    parser.add_argument('image_path', 
                       help='输入图像路径')
    parser.add_argument('--model', 
                       default='./models/Pytorch_Classification_50ep.pt',
                       help='模型文件路径 (默认: ./models/Pytorch_Classification_50ep.pt)')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='显示详细预测信息')
    
    args = parser.parse_args()
    
    # 检查图像文件是否存在
    if not os.path.exists(args.image_path):
        print(f"✗ 图像文件不存在: {args.image_path}")
        sys.exit(1)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"✗ 模型文件不存在: {args.model}")
        sys.exit(1)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(args.model, device)
    
    # 进行预测
    print(f"正在处理图像: {args.image_path}")
    result = predict(args.image_path, model, device)
    
    if result:
        print(f"\n🎯 预测结果:")
        print(f"   类别: {result['class_name']}")
        print(f"   置信度: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        
        if args.verbose:
            print(f"\n📊 详细信息:")
            print(f"   类别ID: {result['class_id']}")
            print(f"   图像路径: {os.path.abspath(args.image_path)}")
            print(f"   模型路径: {os.path.abspath(args.model)}")
            print(f"   使用设备: {device}")
    else:
        print("✗ 预测失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
