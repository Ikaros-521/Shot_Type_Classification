#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - 验证环境和代码设置
"""
import sys
import os

def test_imports():
    """测试必要的库导入"""
    print("🔍 测试库导入...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
    except ImportError as e:
        print(f"❌ TorchVision导入失败: {e}")
        return False
    
    try:
        import PIL
        from PIL import Image
        print(f"✅ Pillow: {PIL.__version__}")
    except ImportError as e:
        print(f"❌ Pillow导入失败: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas: {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas导入失败: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV导入失败: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib导入失败: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn导入失败: {e}")
        return False
    
    return True

def test_file_structure():
    """测试文件结构"""
    print("\n📁 测试文件结构...")
    
    # 检查关键文件
    required_files = [
        'train.py',
        'predict.py', 
        'batch_predict.py',
        'prepare_data.py',
        'requirements.txt',
        'USAGE.md',
        'TRAINING_GUIDE.md'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} 不存在")
    
    # 检查目录
    required_dirs = [
        'models',
        'data', 
        'examples',
        'training'
    ]
    
    for dir in required_dirs:
        if os.path.isdir(dir):
            print(f"✅ {dir}/")
        else:
            print(f"❌ {dir}/ 不存在")

def test_model_file():
    """测试模型文件"""
    print("\n🤖 测试模型文件...")
    
    model_path = './models/Pytorch_Classification_50ep.pt'
    if os.path.exists(model_path):
        print(f"✅ 预训练模型存在: {model_path}")
        
        try:
            import torch
            model_data = torch.load(model_path, map_location='cpu')
            print(f"✅ 模型文件可读，大小: {os.path.getsize(model_path)/1024/1024:.1f} MB")
        except Exception as e:
            print(f"❌ 模型文件读取失败: {e}")
    else:
        print(f"❌ 预训练模型不存在: {model_path}")

def test_example_images():
    """测试示例图像"""
    print("\n🖼️  测试示例图像...")
    
    example_dir = './examples'
    if os.path.exists(example_dir):
        images = [f for f in os.listdir(example_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"✅ 找到 {len(images)} 张示例图像")
        
        # 测试加载一张图像
        if images:
            try:
                from PIL import Image
                img_path = os.path.join(example_dir, images[0])
                img = Image.open(img_path)
                print(f"✅ 示例图像可读: {images[0]} ({img.size})")
            except Exception as e:
                print(f"❌ 示例图像读取失败: {e}")
    else:
        print(f"❌ 示例图像目录不存在: {example_dir}")

def test_script_syntax():
    """测试脚本语法"""
    print("\n📝 测试脚本语法...")
    
    scripts = [
        'train.py',
        'predict.py',
        'batch_predict.py', 
        'prepare_data.py'
    ]
    
    for script in scripts:
        if os.path.exists(script):
            try:
                with open(script, 'r', encoding='utf-8') as f:
                    compile(f.read(), script, 'exec')
                print(f"✅ {script} 语法正确")
            except SyntaxError as e:
                print(f"❌ {script} 语法错误: {e}")
                return False
            except Exception as e:
                print(f"❌ {script} 检查失败: {e}")
        else:
            print(f"❌ {script} 不存在")
    
    return True

def main():
    """主测试函数"""
    print("🚀 镜头类型分类项目 - 环境测试")
    print("=" * 50)
    
    # 运行所有测试
    tests = [
        test_imports,
        test_file_structure,
        test_model_file,
        test_example_images,
        test_script_syntax
    ]
    
    all_passed = True
    for test in tests:
        try:
            result = test()
            if result is False:
                all_passed = False
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有测试通过！环境配置正确。")
        print("\n🎯 现在你可以:")
        print("  • 运行 python train.py 开始训练")
        print("  • 运行 python predict.py image.jpg 进行预测")
        print("  • 运行 python prepare_data.py 准备数据")
    else:
        print("⚠️  部分测试失败，请检查环境配置。")
        print("\n🔧 建议:")
        print("  • 安装缺失的依赖: pip install -r requirements.txt")
        print("  • 检查文件路径和权限")
        print("  • 确保Python版本 >= 3.7")

if __name__ == "__main__":
    main()
