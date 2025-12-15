# 镜头类型分类 - 使用说明
本项目提供了独立的Python脚本来进行图像的镜头类型分类。

## 文件说明

- `predict.py` - 单张图像预测脚本
- `batch_predict.py` - 批量图像预测脚本
- `models/Pytorch_Classification_50ep.pt` - PyTorch训练好的模型

## 环境要求

确保已安装以下依赖：

```bash
pip install torch torchvision pillow
```

## 使用方法

### 1. 单张图像预测

```bash
# 基本用法
python predict.py image.jpg

# 显示详细信息
python predict.py image.jpg --verbose

# 使用自定义模型
python predict.py image.jpg --model custom_model.pt
```

**输出示例：**
```
使用设备: cpu
✓ 成功加载模型: ./models/Pytorch_Classification_50ep.pt
正在处理图像: image.jpg

🎯 预测结果:
   类别: 特写 (Close-up shot, CS)
   置信度: 0.9234 (92.34%)
```

### 2. 批量图像预测

```bash
# 预测文件夹中的所有图像
python batch_predict.py ./images

# 指定输出文件
python batch_predict.py ./images --output my_results.json

# 显示详细处理信息
python batch_predict.py ./images --verbose
```

**输出示例：**
```
使用设备: cpu
✓ 成功加载模型: ./models/Pytorch_Classification_50ep.pt
找到 15 个图像文件
正在处理: image1.jpg
  ✓ 特写 (Close-up shot, CS) (0.923)
正在处理: image2.jpg
  ✓ 远景 (Long shot, LS) (0.876)
...

📊 批量处理完成:
   总计: 15 个文件
   成功: 15 个
   失败: 0 个
✓ 结果已保存到: prediction_results.json

📈 类别分布:
   特写 (Close-up shot, CS): 6 (40.0%)
   远景 (Long shot, LS): 4 (26.7%)
   中景 (Medium shot, MS): 3 (20.0%)
   全景 (Full shot, FS): 2 (13.3%)
```

## 支持的图像格式

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## 分类类别

模型可以将图像分类为以下5种镜头类型：

| 类别ID | 类别名称 | 描述 |
|--------|----------|------|
| 0 | 远景 (Long shot, LS) | 远距离拍摄 |
| 1 | 全景 (Full shot, FS) | 完整的人体 |
| 2 | 中景 (Medium shot, MS) | 膝盖或腰部以上 |
| 3 | 特写 (Close-up shot, CS) | 相对较小的物体，如脸部、手 |
| 4 | 极特写 (Extreme close-up shot, ECS) | 更小的物体部分，如眼睛 |

## 输出格式

### 单张预测输出
- 控制台直接显示预测结果
- 包含类别名称和置信度

### 批量预测输出
- 生成JSON文件，包含：
  - 处理时间戳
  - 总图像数量
  - 类别分布统计
  - 每张图像的详细预测结果

**JSON输出示例：**
```json
{
  "timestamp": "2025-12-15T10:30:45.123456",
  "total_images": 15,
  "class_distribution": {
    "特写 (Close-up shot, CS)": 6,
    "远景 (Long shot, LS)": 4,
    "中景 (Medium shot, MS)": 3,
    "全景 (Full shot, FS)": 2
  },
  "results": [
    {
      "image_path": "image1.jpg",
      "filename": "image1.jpg",
      "class_id": 3,
      "class_name": "特写 (Close-up shot, CS)",
      "confidence": 0.9234
    }
  ]
}
```

## 注意事项

1. **模型路径**：确保模型文件 `Pytorch_Classification_50ep.pt` 存在于 `models/` 目录中
2. **图像质量**：输入图像应该清晰，包含明显的视觉特征
3. **GPU支持**：如果有CUDA GPU，脚本会自动使用GPU加速
4. **内存使用**：批量处理大量图像时注意内存使用情况

## 故障排除

### 常见错误及解决方案：

1. **模型文件不存在**
   ```
   ✗ 模型文件不存在: ./models/Pytorch_Classification_50ep.pt
   ```
   解决：确保模型文件在正确路径

2. **图像文件格式不支持**
   ```
   ✗ 图像处理失败: cannot identify image file
   ```
   解决：使用支持的图像格式（JPG、PNG、BMP、TIFF）

3. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决：使用CPU模式或减少批量处理数量

## 性能优化建议

1. **GPU加速**：使用NVIDIA GPU可显著提升处理速度
2. **批量处理**：对于大量图像，使用批量预测脚本
3. **图像预处理**：确保图像分辨率适中，避免过大的图像文件

## 技术细节

- **模型架构**：基于MobileNetV3
- **输入尺寸**：224×224 像素
- **预处理**：ImageNet标准化
- **置信度阈值**：无固定阈值，显示所有预测概率
