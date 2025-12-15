# 镜头类型分类 - 训练指南
本指南将详细介绍如何从头开始训练镜头类型分类模型。

## 目录

1. [环境准备](#环境准备)
2. [数据准备](#数据准备)
3. [模型训练](#模型训练)
4. [模型评估](#模型评估)
5. [调优技巧](#调优技巧)
6. [故障排除](#故障排除)

## 环境准备

### 1. 系统要求

- Python 3.7+
- CUDA 10.2+ (可选，用于GPU加速)
- 至少8GB内存
- 至少20GB可用磁盘空间

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv shot_classification_env
source shot_classification_env/bin/activate  # Linux/Mac
# shot_classification_env\Scripts\activate  # Windows

# 安装PyTorch (根据你的CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install pandas numpy matplotlib seaborn scikit-learn pillow opencv-python tqdm
```

### 3. 下载MovieShots数据集

从以下网址下载MovieShots数据集：
- https://paperswithcode.com/dataset/movieshots
- https://arxiv.org/abs/2008.03548

下载后，将数据集文件组织如下：

```
data/
├── v1_full_trailer.json          # 标注文件
└── trailer/                      # 视频文件
    ├── tt0012345/                # 电影文件夹
    │   ├── shot_0001.mp4
    │   ├── shot_0002.mp4
    │   └── ...
    └── ...
```

## 数据准备

### 1. 自动化数据准备

使用我们提供的自动化脚本：

```bash
# 执行完整的数据准备流程
python prepare_data.py --data-dir ./data --full-process

# 或者分步骤执行
python prepare_data.py --data-dir ./data --parse-json
python prepare_data.py --data-dir ./data --organize-videos  
python prepare_data.py --data-dir ./data --extract-frames
python prepare_data.py --data-dir ./data --split-data
```

### 2. 手动数据准备

如果你想手动控制每个步骤：

#### 步骤1: 解析JSON标签
```bash
python prepare_data.py --data-dir ./data --parse-json
```

这会生成 `dataset.csv` 文件，包含所有视频的标签信息。

#### 步骤2: 组织视频文件
```bash
python prepare_data.py --data-dir ./data --organize-videos
```

将视频按类别组织到不同文件夹：
```
data/categorized/
├── CS/     # 特写镜头
├── ECS/    # 极特写镜头
├── FS/     # 全景镜头
├── LS/     # 远景镜头
└── MS/     # 中景镜头
```

#### 步骤3: 提取视频帧
```bash
# 默认每25帧提取一帧
python prepare_data.py --data-dir ./data --extract-frames

# 自定义帧间隔
python prepare_data.py --data-dir ./data --extract-frames --frame-interval 30
```

#### 步骤4: 分割训练集和测试集
```bash
# 默认20%作为测试集
python prepare_data.py --data-dir ./data --split-data

# 自定义测试集比例
python prepare_data.py --data-dir ./data --split-data --test-ratio 0.3
```

### 3. 数据结构说明

最终的数据结构如下：

```
data/frames/
├── training/                    # 训练集 (80%)
│   ├── CS/                     # 特写镜头
│   ├── ECS/                    # 极特写镜头
│   ├── FS/                     # 全景镜头
│   ├── LS/                     # 远景镜头
│   └── MS/                     # 中景镜头
└── testing/                     # 测试集 (20%)
    ├── CS/
    ├── ECS/
    ├── FS/
    ├── LS/
    └── MS/
```

## 模型训练

### 1. 基础训练

```bash
# 基础训练命令
python train.py --data-dir ./data/frames/training

# 指定训练轮数和批次大小
python train.py --data-dir ./data/frames/training --epochs 100 --batch-size 32

# 自定义学习率
python train.py --data-dir ./data/frames/training --lr 0.0005
```

### 2. 高级训练选项

```bash
# 完整参数示例
python train.py \
    --data-dir ./data/frames/training \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001 \
    --val-split 0.2 \
    --model-name shot_classifier_v1 \
    --output-dir ./models
```

### 3. 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | 必需 | 训练数据目录 |
| `--epochs` | 50 | 训练轮数 |
| `--batch-size` | 16 | 批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--val-split` | 0.2 | 验证集比例 |
| `--model-name` | 时间戳 | 模型保存名称 |
| `--output-dir` | ./models | 模型输出目录 |

### 4. 训练过程监控

训练过程中会显示：

```
Epoch 1/50 [训练]: 100%|██████████| 245/245 [02:15<00:00, Loss: 0.8234, Acc: 0.7123]
Epoch 1/50 [验证]: 100%|██████████| 62/62 [00:15<00:00, Loss: 0.6234, Acc: 0.8456]
Epoch 1/50 (150.2s) - Train Loss: 0.8234, Train Acc: 0.7123, Val Loss: 0.6234, Val Acc: 0.8456
```

## 模型评估

### 1. 自动评估

训练完成后会自动生成评估报告：

```
📈 分类报告:
  极特写 (Extreme close-up shot):
    精确率: 0.9234, 召回率: 0.9123, F1分数: 0.9178
  特写 (Close-up shot):
    精确率: 0.8956, 召回率: 0.9345, F1分数: 0.9146
  ...
  总体准确率: 0.9123
```

### 2. 可视化结果

训练过程中会生成：

- **训练历史图**: 显示损失和准确率的变化
- **混淆矩阵**: 显示各类别的预测性能

### 3. 模型文件

训练完成后会生成以下文件：

```
models/
├── shot_classifier_20231215_143022.pt           # 模型权重
├── shot_classifier_20231215_143022_history.pth  # 训练历史
└── shot_classifier_20231215_143022_report.json  # 评估报告
```

## 调优技巧

### 1. 超参数调优

#### 学习率调整
```bash
# 较小的学习率，更稳定
python train.py --data-dir ./data/frames/training --lr 0.0001

# 使用学习率调度
python train.py --data-dir ./data/frames/training --lr 0.01
```

#### 批次大小调整
```bash
# 大批次大小（需要更多内存）
python train.py --data-dir ./data/frames/training --batch-size 32

# 小批次大小（内存较少时）
python train.py --data-dir ./data/frames/training --batch-size 8
```

#### 训练轮数
```bash
# 更多的训练轮数
python train.py --data-dir ./data/frames/training --epochs 100

# 早停机制（观察验证损失）
python train.py --data-dir ./data/frames/training --epochs 200
```

### 2. 数据增强调整

在 `train.py` 中的 `get_data_transforms()` 函数中调整：

```python
# 增强数据增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),           # 水平翻转
    transforms.RandomRotation(degrees=15),            # 旋转
    transforms.ColorJitter(brightness=0.3,            # 亮度调整
                          contrast=0.3,              # 对比度调整
                          saturation=0.3,             # 饱和度调整
                          hue=0.1),                  # 色调调整
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # 随机裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])
```

### 3. 模型架构调整

#### 更换基础模型
```python
# 在 create_model() 函数中
# 使用不同的预训练模型
model = models.resnet50(pretrained=True)           # ResNet-50
model = models.efficientnet_b0(pretrained=True)    # EfficientNet-B0
model = models.vgg16(pretrained=True)               # VGG-16

# 修改分类器
if hasattr(model, 'fc'):  # ResNet
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif hasattr(model, 'classifier'):  # VGG, MobileNet
    if isinstance(model.classifier, nn.Sequential):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
```

### 4. 正则化技术

#### Dropout调整
```python
# 在模型中添加dropout
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[0].in_features, 1280),
    nn.Hardswish(),
    nn.Dropout(p=0.3),  # 调整dropout率
    nn.Linear(1280, num_classes)
)
```

#### 权重衰减
```python
# 在训练命令中调整
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)  # 增加权重衰减
```

## 故障排除

### 1. 常见错误

#### CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案:**
- 减小批次大小: `--batch-size 8`
- 使用CPU训练: 设置 `device = 'cpu'`
- 清理GPU缓存: `torch.cuda.empty_cache()`

#### 数据目录不存在
```
✗ 数据目录不存在: ./data/frames/training
```
**解决方案:**
- 确保已完成数据准备: `python prepare_data.py --full-process`
- 检查路径是否正确

#### 图像加载失败
```
✗ 图像处理失败: cannot identify image file
```
**解决方案:**
- 检查图像文件是否损坏
- 确保图像格式正确（JPG、PNG）
- 重新提取视频帧

### 2. 性能问题

#### 训练速度慢
**可能原因和解决方案:**
- **CPU训练**: 使用GPU加速
- **批次大小过小**: 适当增加批次大小
- **数据加载瓶颈**: 增加 `num_workers` 参数
- **图像过大**: 调整图像尺寸

#### 过拟合
**表现:**
- 训练准确率高，验证准确率低
- 训练损失持续下降，验证损失开始上升

**解决方案:**
- 增加数据增强
- 使用dropout
- 增加权重衰减
- 减少模型复杂度
- 增加训练数据

#### 欠拟合
**表现:**
- 训练和验证准确率都较低
- 损失值较高且不下降

**解决方案:**
- 增加模型复杂度
- 减少正则化
- 增加训练轮数
- 调整学习率

### 3. 调试技巧

#### 检查数据加载
```python
# 添加到训练脚本中调试
dataset = ShotDataset(args.data_dir, train_transform)
print(f"数据集大小: {len(dataset)}")
print(f"类别分布: {[sum(1 for _, label in dataset if label == i) for i in range(5)]}")

# 检查单个样本
image, label = dataset[0]
print(f"图像形状: {image.shape}")
print(f"标签: {label}")
```

#### 可视化数据
```python
import matplotlib.pyplot as plt

# 显示一些训练样本
def show_samples(dataset, num_samples=6):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, (image, label) in enumerate(dataset):
        if i >= num_samples:
            break
        ax = axes[i//3, i%3]
        # 反归一化
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image = torch.clamp(image, 0, 1)
        ax.imshow(image.permute(1, 2, 0))
        ax.set_title(f'Class: {CLASS_NAMES[label]}')
        ax.axis('off')
    plt.show()

show_samples(train_dataset)
```

## 最佳实践

### 1. 实验管理
- 为每次实验使用不同的模型名称
- 保存训练历史和超参数
- 使用版本控制管理代码

### 2. 模型选择
- 基于验证集准确率选择最佳模型
- 考虑模型大小和推理速度的平衡
- 保存多个检查点以备后用

### 3. 持续改进
- 定期重新评估模型性能
- 根据新数据更新模型
- 尝试不同的架构和技术

## 进阶主题

### 1. 迁移学习
```python
# 使用预训练模型进行特征提取
model = models.mobilenet_v3_large(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # 冻结特征提取层

# 只训练分类器
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
```

### 2. 集成学习
```python
# 训练多个模型并集成预测
models_ensemble = [create_model() for _ in range(5)]
# ... 训练每个模型 ...

def ensemble_predict(models, input):
    outputs = [model(input) for model in models]
    avg_output = torch.mean(torch.stack(outputs), dim=0)
    return avg_output
```

### 3. 自动超参数优化
使用Optuna等工具进行自动超参数搜索：

```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    # ... 训练模型并返回验证准确率 ...
    return validation_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## 总结

通过本指南，你应该能够：

1. ✅ 准备MovieShots数据集
2. ✅ 训练高性能的镜头分类模型
3. ✅ 评估和优化模型性能
4. ✅ 解决常见训练问题

记住，深度学习是一个迭代的过程。不断实验、调整和改进是获得最佳结果的关键。
