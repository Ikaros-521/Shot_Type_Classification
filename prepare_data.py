#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备脚本
从原始MovieShots数据集准备训练数据
"""
import os
import sys
import json
import pandas as pd
import cv2
import argparse
import random
from pathlib import Path
from tqdm import tqdm

def parse_json_to_csv(json_path, output_csv):
    """解析JSON数据到CSV格式"""
    print(f"📖 解析JSON文件: {json_path}")
    
    df = pd.DataFrame()
    i = 0
    
    with open(json_path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                print(f"✓ 解析完成，共处理 {i} 个文件夹")
                break
            
            # 查找文件夹标识
            if "tt" in line:
                i += 1
                folder_name = line.translate({ord(c): None for c in '\\n\'\" :\{\}\n'})
                
                # 查找文件
                line = f.readline()
                while "tt" not in line:
                    if "\"00" in line:
                        file_name = line.translate({ord(c): None for c in '\\n\'\" :\{\}\n'})
                        line = f.readline()
                        label = line.replace('\"label": ', '')
                        label = label.translate({ord(c): None for c in '\\n\'\" :\{\}\n\,'})
                        
                        df = df.append({
                            'Folder': folder_name, 
                            'FileName': file_name, 
                            'Label': label
                        }, ignore_index=True)
                    
                    line = f.readline()
                    if not line:
                        break
    
    # 保存CSV
    df.to_csv(output_csv, index=False)
    print(f"✓ 数据已保存到: {output_csv}")
    print(f"📊 总计 {len(df)} 个数据样本")
    
    # 显示类别分布
    print("\n📈 类别分布:")
    label_counts = df['Label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    return df

def organize_videos(df, source_dir, target_dir):
    """组织视频文件到分类文件夹"""
    print(f"🎬 组织视频文件...")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    
    # 创建目标目录
    labels = ['CS', 'ECS', 'FS', 'LS', 'MS']
    for label in labels:
        label_dir = os.path.join(target_dir, label)
        os.makedirs(label_dir, exist_ok=True)
    
    moved_count = 0
    missing_count = 0
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="处理视频文件"):
        filename = f'shot_{row["FileName"]}.mp4'
        old_file_path = os.path.join(source_dir, row['Folder'], filename)
        
        if os.path.isfile(old_file_path):
            new_filename = f'{row["Folder"]}_{filename}'
            new_file_path = os.path.join(target_dir, row['Label'], new_filename)
            
            try:
                os.rename(old_file_path, new_file_path)
                moved_count += 1
            except Exception as e:
                print(f"✗ 移动文件失败: {old_file_path} -> {new_file_path}, 错误: {e}")
                missing_count += 1
        else:
            print(f"✗ 文件不存在: {old_file_path}")
            missing_count += 1
    
    print(f"\n📊 视频文件组织完成:")
    print(f"  成功移动: {moved_count}")
    print(f"  缺失文件: {missing_count}")

def extract_frames_from_videos(video_dir, output_dir, frame_interval=25):
    """从视频提取帧图像"""
    print(f"🖼️  从视频提取帧图像...")
    print(f"视频目录: {video_dir}")
    print(f"输出目录: {output_dir}")
    print(f"帧间隔: {frame_interval}")
    
    # 创建输出目录
    labels = ['CS', 'ECS', 'FS', 'LS', 'MS']
    for label in labels:
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
    
    total_extracted = 0
    processed_videos = 0
    
    for label in labels:
        video_label_dir = os.path.join(video_dir, label)
        if not os.path.exists(video_label_dir):
            print(f"✗ 视频目录不存在: {video_label_dir}")
            continue
        
        video_files = [f for f in os.listdir(video_label_dir) if f.lower().endswith('.mp4')]
        print(f"\n处理 {label} 类别，共 {len(video_files)} 个视频")
        
        for video_file in tqdm(video_files, desc=f"提取{label}帧"):
            video_path = os.path.join(video_label_dir, video_file)
            
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"✗ 无法打开视频: {video_path}")
                    continue
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count = 0
                
                for frame_idx in range(0, total_frames, frame_interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        output_filename = f"{os.path.splitext(video_file)[0]}_frame_{frame_idx}.jpg"
                        output_path = os.path.join(output_dir, label, output_filename)
                        cv2.imwrite(output_path, frame)
                        frame_count += 1
                
                cap.release()
                total_extracted += frame_count
                processed_videos += 1
                
            except Exception as e:
                print(f"✗ 处理视频失败: {video_path}, 错误: {e}")
    
    print(f"\n📊 帧提取完成:")
    print(f"  处理视频数: {processed_videos}")
    print(f"  提取帧数: {total_extracted}")

def split_train_test_data(frame_dir, train_dir, test_dir, test_ratio=0.2, random_seed=42):
    """分割训练集和测试集"""
    print(f"🔀 分割训练集和测试集...")
    print(f"源目录: {frame_dir}")
    print(f"训练集目录: {train_dir}")
    print(f"测试集目录: {test_dir}")
    print(f"测试集比例: {test_ratio}")
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 创建目标目录
    labels = ['CS', 'ECS', 'FS', 'LS', 'MS']
    for label in labels:
        for target_dir in [train_dir, test_dir]:
            label_dir = os.path.join(target_dir, label)
            os.makedirs(label_dir, exist_ok=True)
    
    total_files = 0
    train_files = 0
    test_files = 0
    
    for label in labels:
        source_label_dir = os.path.join(frame_dir, label)
        if not os.path.exists(source_label_dir):
            print(f"✗ 源目录不存在: {source_label_dir}")
            continue
        
        image_files = [f for f in os.listdir(source_label_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        random.shuffle(image_files)
        
        split_index = int(len(image_files) * (1 - test_ratio))
        train_list = image_files[:split_index]
        test_list = image_files[split_index:]
        
        # 移动训练文件
        for img_file in train_list:
            src = os.path.join(source_label_dir, img_file)
            dst = os.path.join(train_dir, label, img_file)
            try:
                os.rename(src, dst)
                train_files += 1
            except Exception as e:
                print(f"✗ 移动训练文件失败: {src} -> {dst}, 错误: {e}")
        
        # 移动测试文件
        for img_file in test_list:
            src = os.path.join(source_label_dir, img_file)
            dst = os.path.join(test_dir, label, img_file)
            try:
                os.rename(src, dst)
                test_files += 1
            except Exception as e:
                print(f"✗ 移动测试文件失败: {src} -> {dst}, 错误: {e}")
        
        total_files += len(image_files)
        print(f"  {label}: 总计 {len(image_files)}, 训练 {len(train_list)}, 测试 {len(test_list)}")
    
    print(f"\n📊 数据分割完成:")
    print(f"  总文件数: {total_files}")
    print(f"  训练文件数: {train_files} ({train_files/total_files*100:.1f}%)")
    print(f"  测试文件数: {test_files} ({test_files/total_files*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(
        description='MovieShots数据集准备工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整流程
  python prepare_data.py --data-dir ./data --full-process
  
  # 仅解析JSON到CSV
  python prepare_data.py --data-dir ./data --parse-json
  
  # 仅组织视频文件
  python prepare_data.py --data-dir ./data --organize-videos
  
  # 仅提取视频帧
  python prepare_data.py --data-dir ./data --extract-frames --frame-interval 30
  
  # 仅分割数据集
  python prepare_data.py --data-dir ./data --split-data --test-ratio 0.3
        """
    )
    
    parser.add_argument('--data-dir',
                       default='./data',
                       help='数据根目录 (默认: ./data)')
    
    parser.add_argument('--full-process',
                       action='store_true',
                       help='执行完整的数据准备流程')
    
    parser.add_argument('--parse-json',
                       action='store_true',
                       help='解析JSON到CSV')
    
    parser.add_argument('--organize-videos',
                       action='store_true',
                       help='组织视频文件')
    
    parser.add_argument('--extract-frames',
                       action='store_true',
                       help='从视频提取帧')
    
    parser.add_argument('--split-data',
                       action='store_true',
                       help='分割训练集和测试集')
    
    parser.add_argument('--frame-interval',
                       type=int,
                       default=25,
                       help='视频帧提取间隔 (默认: 25)')
    
    parser.add_argument('--test-ratio',
                       type=float,
                       default=0.2,
                       help='测试集比例 (默认: 0.2)')
    
    args = parser.parse_args()
    
    # 设置路径
    json_path = os.path.join(args.data_dir, 'v1_full_trailer.json')
    csv_path = os.path.join(args.data_dir, 'dataset.csv')
    trailer_dir = os.path.join(args.data_dir, 'trailer')
    categorized_dir = os.path.join(args.data_dir, 'categorized')
    frames_dir = os.path.join(args.data_dir, 'frames')
    train_dir = os.path.join(args.data_dir, 'frames', 'training')
    test_dir = os.path.join(args.data_dir, 'frames', 'testing')
    
    print("🚀 MovieShots数据集准备工具")
    print("=" * 50)
    
    # 如果没有指定具体步骤，执行完整流程
    if not any([args.parse_json, args.organize_videos, args.extract_frames, args.split_data]):
        args.full_process = True
    
    try:
        # 步骤1: 解析JSON到CSV
        if args.full_process or args.parse_json:
            if not os.path.exists(json_path):
                print(f"✗ JSON文件不存在: {json_path}")
                print("请确保已下载MovieShots数据集并将v1_full_trailer.json放在data目录中")
                sys.exit(1)
            
            df = parse_json_to_csv(json_path, csv_path)
        
        # 步骤2: 组织视频文件
        if args.full_process or args.organize_videos:
            if 'df' not in locals():
                df = pd.read_csv(csv_path)
            
            if not os.path.exists(trailer_dir):
                print(f"✗ 视频源目录不存在: {trailer_dir}")
                print("请确保已下载MovieShots数据集的视频文件")
                sys.exit(1)
            
            organize_videos(df, trailer_dir, categorized_dir)
        
        # 步骤3: 提取视频帧
        if args.full_process or args.extract_frames:
            if not os.path.exists(categorized_dir):
                print(f"✗ 分类视频目录不存在: {categorized_dir}")
                print("请先执行视频组织步骤")
                sys.exit(1)
            
            extract_frames_from_videos(categorized_dir, frames_dir, args.frame_interval)
        
        # 步骤4: 分割训练集和测试集
        if args.full_process or args.split_data:
            if not os.path.exists(frames_dir):
                print(f"✗ 帧图像目录不存在: {frames_dir}")
                print("请先执行帧提取步骤")
                sys.exit(1)
            
            split_train_test_data(frames_dir, train_dir, test_dir, args.test_ratio)
        
        print("\n🎉 数据准备完成！")
        print("\n📁 生成的目录结构:")
        print(f"{args.data_dir}/")
        print("├── dataset.csv                    # 数据标签文件")
        print("├── categorized/                   # 分类后的视频文件")
        print("│   ├── CS/")
        print("│   ├── ECS/")
        print("│   ├── FS/")
        print("│   ├── LS/")
        print("│   └── MS/")
        print("└── frames/                       # 提取的帧图像")
        print("    ├── training/                 # 训练集")
        print("    │   ├── CS/")
        print("    │   ├── ECS/")
        print("    │   ├── FS/")
        print("    │   ├── LS/")
        print("    │   └── MS/")
        print("    └── testing/                  # 测试集")
        print("        ├── CS/")
        print("        ├── ECS/")
        print("        ├── FS/")
        print("        ├── LS/")
        print("        └── MS/")
        
        print(f"\n🎯 现在可以使用以下命令训练模型:")
        print(f"python train.py --data-dir {train_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
