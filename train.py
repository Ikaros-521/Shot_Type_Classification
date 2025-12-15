#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
镜头类型分类模型训练脚本
基于PyTorch的深度学习模型训练
"""
import os
import sys
import argparse
import time
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

warnings.simplefilter("ignore")

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 类别映射
CLASS_NAMES = ['ECS', 'CS', 'FS', 'LS', 'MS']  # 按字母顺序排序
CLASS_MAPPING = {
    'ECS': '极特写 (Extreme close-up shot)',
    'CS': '特写 (Close-up shot)', 
    'FS': '全景 (Full shot)',
    'LS': '远景 (Long shot)',
    'MS': '中景 (Medium shot)'
}

class ShotDataset(torch.utils.data.Dataset):
    """自定义数据集类"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 加载数据
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_transforms():
    """获取数据预处理变换"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(num_classes=5):
    """创建模型架构"""
    # 使用MobileNetV3作为基础模型
    model = models.mobilenet_v3_large(pretrained=True)
    
    # 修改分类器
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, scheduler=None):
    """训练模型"""
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    print(f"开始训练，共 {num_epochs} 个epoch")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [训练]')
        
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            
            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{running_corrects.double() / (train_pbar.n * train_loader.batch_size):.4f}'
            })
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [验证]')
            
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{val_corrects.double() / (val_pbar.n * val_loader.batch_size):.4f}'
                })
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_corrects.double() / len(val_loader.dataset)
        
        # 学习率调度
        if scheduler:
            scheduler.step()
        
        # 保存最佳模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_state = model.state_dict()
        
        # 记录历史
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc.item())
        val_accs.append(epoch_val_acc.item())
        
        # 打印epoch结果
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        
        if (epoch + 1) % 10 == 0:
            print(f'最佳验证准确率: {best_val_acc:.4f}')
    
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    return model, history

def evaluate_model(model, test_loader, class_names):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    
    print("正在评估模型...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='评估中'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 生成分类报告
    report = classification_report(all_labels, all_preds, 
                                  target_names=class_names,
                                  output_dict=True)
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    return report, cm, all_preds, all_labels

def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(history['train_losses'], label='训练损失', color='blue')
    ax1.plot(history['val_losses'], label='验证损失', color='red')
    ax1.set_title('模型损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(history['train_accs'], label='训练准确率', color='blue')
    ax2.plot(history['val_accs'], label='验证准确率', color='red')
    ax2.set_title('模型准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存到: {save_path}")
    
    plt.show()

def plot_confusion_matrix(cm, class_names, save_path=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    
    # 转换为中文类别名称
    chinese_names = [CLASS_MAPPING[name] for name in class_names]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=chinese_names,
                yticklabels=chinese_names)
    
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    
    plt.show()

def save_model(model, save_path, history=None, evaluation_report=None):
    """保存模型和相关信息"""
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")
    
    # 保存训练历史
    if history:
        history_path = save_path.replace('.pt', '_history.pth')
        torch.save(history, history_path)
        print(f"训练历史已保存到: {history_path}")
    
    # 保存评估报告
    if evaluation_report:
        import json
        report_path = save_path.replace('.pt', '_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
        print(f"评估报告已保存到: {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description='镜头类型分类模型训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python train.py --data-dir ./data/frames/training
  python train.py --data-dir ./data/frames/training --epochs 100 --batch-size 32
  python train.py --data-dir ./data/frames/training --lr 0.001 --model-name my_model
        """
    )
    
    parser.add_argument('--data-dir', 
                       required=True,
                       help='训练数据目录路径')
    parser.add_argument('--epochs', 
                       type=int, 
                       default=50,
                       help='训练轮数 (默认: 50)')
    parser.add_argument('--batch-size', 
                       type=int, 
                       default=16,
                       help='批次大小 (默认: 16)')
    parser.add_argument('--lr', 
                       type=float, 
                       default=0.001,
                       help='学习率 (默认: 0.001)')
    parser.add_argument('--val-split', 
                       type=float, 
                       default=0.2,
                       help='验证集比例 (默认: 0.2)')
    parser.add_argument('--model-name',
                       default=f'shot_classifier_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='模型保存名称')
    parser.add_argument('--output-dir',
                       default='./models',
                       help='模型输出目录 (默认: ./models)')
    parser.add_argument('--no-save',
                       action='store_true',
                       help='不保存模型')
    parser.add_argument('--no-plot',
                       action='store_true',
                       help='不显示图表')
    
    args = parser.parse_args()
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"✗ 数据目录不存在: {args.data_dir}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 开始训练镜头类型分类模型")
    print(f"📁 数据目录: {args.data_dir}")
    print(f"🎯 训练轮数: {args.epochs}")
    print(f"📦 批次大小: {args.batch_size}")
    print(f"📈 学习率: {args.lr}")
    print(f"🔢 验证集比例: {args.val_split}")
    print("-" * 50)
    
    # 获取数据变换
    train_transform, val_transform = get_data_transforms()
    
    # 创建数据集
    full_dataset = ShotDataset(args.data_dir, train_transform)
    print(f"📊 总样本数: {len(full_dataset)}")
    
    # 分割训练集和验证集
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 为验证集设置正确的变换
    val_dataset.dataset.transform = val_transform
    
    print(f"📚 训练集样本数: {len(train_dataset)}")
    print(f"🧪 验证集样本数: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, 
                             batch_size=args.batch_size, 
                             shuffle=True, 
                             num_workers=4)
    val_loader = DataLoader(val_dataset, 
                           batch_size=args.batch_size, 
                           shuffle=False, 
                           num_workers=4)
    
    # 创建模型
    model = create_model(len(CLASS_NAMES))
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # 开始训练
    start_time = time.time()
    trained_model, history = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer, args.epochs, scheduler
    )
    training_time = time.time() - start_time
    
    print(f"\n🎉 训练完成！总耗时: {training_time/60:.2f} 分钟")
    
    # 评估模型
    print("\n📊 评估模型性能...")
    val_loader_for_eval = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    report, cm, preds, labels = evaluate_model(trained_model, val_loader_for_eval, CLASS_NAMES)
    
    # 打印评估结果
    print("\n📈 分类报告:")
    for class_name in CLASS_NAMES:
        chinese_name = CLASS_MAPPING[class_name]
        precision = report[class_name]['precision']
        recall = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        print(f"  {chinese_name}:")
        print(f"    精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
    
    print(f"  总体准确率: {report['accuracy']:.4f}")
    
    # 保存模型
    if not args.no_save:
        model_path = os.path.join(args.output_dir, f"{args.model_name}.pt")
        save_model(trained_model, model_path, history, report)
    
    # 绘制图表
    if not args.no_plot:
        plot_training_history(history)
        plot_confusion_matrix(cm, CLASS_NAMES)

if __name__ == "__main__":
    main()
