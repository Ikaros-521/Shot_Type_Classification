#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量镜头类型分类预测脚本
对文件夹中的所有图像进行批量预测
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import sys
import glob
import json
from datetime import datetime

# 导入单张预测的函数
from predict import load_model, preprocess_image, predict, CLASS_MAPPING

def batch_predict(input_dir, output_file, model, device, verbose=False):
    """批量预测文件夹中的图像"""
    
    # 支持的图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # 查找所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"✗ 在目录 {input_dir} 中未找到图像文件")
        return []
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    results = []
    processed = 0
    failed = 0
    
    for image_path in image_files:
        try:
            if verbose:
                print(f"正在处理: {os.path.basename(image_path)}")
            
            result = predict(image_path, model, device)
            if result:
                result['image_path'] = image_path
                result['filename'] = os.path.basename(image_path)
                results.append(result)
                processed += 1
                
                if verbose:
                    print(f"  ✓ {result['class_name']} ({result['confidence']:.3f})")
            else:
                failed += 1
                print(f"  ✗ 预测失败")
                
        except Exception as e:
            failed += 1
            print(f"  ✗ 处理失败: {e}")
    
    print(f"\n📊 批量处理完成:")
    print(f"   总计: {len(image_files)} 个文件")
    print(f"   成功: {processed} 个")
    print(f"   失败: {failed} 个")
    
    # 保存结果
    if results:
        save_results(results, output_file)
    
    return results

def save_results(results, output_file):
    """保存预测结果到文件"""
    
    # 准备输出数据
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'total_images': len(results),
        'class_distribution': {},
        'results': results
    }
    
    # 统计类别分布
    for result in results:
        class_name = result['class_name']
        if class_name not in output_data['class_distribution']:
            output_data['class_distribution'][class_name] = 0
        output_data['class_distribution'][class_name] += 1
    
    # 保存为JSON格式
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 结果已保存到: {output_file}")
    except Exception as e:
        print(f"✗ 保存结果失败: {e}")
    
    # 打印类别分布统计
    print(f"\n📈 类别分布:")
    for class_name, count in output_data['class_distribution'].items():
        percentage = (count / len(results)) * 100
        print(f"   {class_name}: {count} ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(
        description='批量镜头类型分类预测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python batch_predict.py ./images                     # 预测images文件夹中的所有图像
  python batch_predict.py ./images --output results.json  # 指定输出文件
  python batch_predict.py ./images --verbose           # 显示详细信息
        """
    )
    
    parser.add_argument('input_dir', 
                       help='输入图像目录路径')
    parser.add_argument('--output', '-o',
                       default='prediction_results.json',
                       help='输出结果文件路径 (默认: prediction_results.json)')
    parser.add_argument('--model', 
                       default='./models/Pytorch_Classification_50ep.pt',
                       help='模型文件路径 (默认: ./models/Pytorch_Classification_50ep.pt)')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='显示详细处理信息')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.isdir(args.input_dir):
        print(f"✗ 输入目录不存在: {args.input_dir}")
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
    
    # 批量预测
    print(f"开始批量处理目录: {args.input_dir}")
    results = batch_predict(args.input_dir, args.output, model, device, args.verbose)

if __name__ == "__main__":
    main()
