# 镜头类型分类
本项目提供两个模型，一个是PyTorch实现，准确率0.90，训练了50个epoch；另一个是TensorFlow实现，准确率0.89，训练了12个epoch。准确率基于从训练集中分离出的测试集。
请勿将其用于商业目的，因为数据集不允许这样做。

## 模型输入/输出

训练好的模型位于models文件夹中。您可以使用`PyTorch_Model_Classifier.ipynb`测试PyTorch实现，使用`TF_Model_Classifier.ipynb`测试TensorFlow实现。

两个模型都将图像（或视频帧）作为输入，并输出以下5个类别之一的镜头类型：

| 类别                        | 描述                                   |
|------------------------------|-----------------------------------------------|
| 远景 (Long shot, LS)               | 远距离拍摄。                              |
| 全景 (Full shot, FS)               | 完整的人体。                           | 
| 中景 (Medium shot, MS)             | 膝盖或腰部以上。                            |
| 特写 (Close-up shot, CS)           | 相对较小的物体，如脸部、手。  |
| 极特写 (Extreme close-up shot, ECS)  | 更小的物体部分，如眼睛    |

## 各类别示例

<div align="center">
  <table border="0" bgcolor="#000000">
      <tr>
        <td valign="top" align="center"><img src="/examples/1.jpg" width="50%"></img> <br />特写镜头 (CS) 示例</td>
        <td valign="top" align="center"> <img src="/examples/2.jpg" width="50%"></img> <br />极特写镜头 (ECS) 示例 </td>
        <td valign="top" align="center"> <img src="/examples/3.jpg" width="50%"></img> <br />全景镜头 (FS) 示例 </td>
      </tr>
    </table>
    
  <table border="0" bgcolor="#000000">
      <tr>
        <td valign="top" align="center"><img src="/examples/4.jpg" width="50%"></img><br /> 远景镜头 (LS) 示例</td>
        <td valign="top" align="center"> <img src="/examples/5.jpg" width="50%"></img><br /> 中景镜头 (MS) 示例 </td>
      </tr>
    </table>
</div>

## 环境要求

### PyTorch实现所需依赖

```
PyTorch 
Pillow
numpy
torchvision
```

### TensorFlow实现所需依赖

```
tensorflow
OpenCV
numpy
```

## 数据集

两个模型都基于MovieShots数据集进行训练：
https://paperswithcode.com/dataset/movieshots 
https://arxiv.org/abs/2008.03548

## 模型性能

数据被随机分割为60%（训练集）、20%（验证集）和20%（测试集），报告的数据基于20%的测试集。

### PyTorch实现的性能如下：

|     |精确率|召回率|F1分数|支持数|
|-----|---------|------|--------|-------|
|CS       |0.90|      0.87|      0.88|       692|
|ECS      |0.89|      0.91|      0.90|       636|
|FS       |0.93|      0.90|      0.92|       623|
|LS       |0.91|      0.97|      0.94|       617|
|MS       |0.92|      0.90|      0.91|       776|
|准确率 |    |          |      0.91|      3344|
|宏平均|0.91|      0.91|      0.91|      3344|
|加权平均   |0.91|      0.91|      0.91|      3344|

### TensorFlow模型的性能如下：

|     |精确率|召回率|F1分数|支持数|
|-----|---------|------|--------|-------|
|CS       |0.82      |0.88      |0.85       |877|
|ECS      |0.92      |0.84      |0.88       |846|
|FS       |0.89      |0.91      |0.90       |793|
|LS       |0.89      |0.96      |0.92       |738|
|MS       |0.91      |0.85      |0.88       |924|
|准确率 |          |          |0.88       |4178|
|宏平均|0.89      |0.89       |0.89      |4178|
|加权平均   |0.89      |0.88       |0.88      |4178|

## 模型训练

训练模型的代码可在training文件夹中找到。

重新训练模型的步骤：
1. 下载数据集
2. 按照`DataSet_CleanUp.ipynb`中的步骤操作
3. 使用PyTorch或TensorFlow实现文件训练模型

PyTorch实现基于https://www.kaggle.com/code/oknashar/brain-tumor-detection-using-pytorch?scriptVersionId=90753009&cellId=15中的MRI肿瘤检测，并使用mobilenet_v3。

TensorFlow实现基于https://www.kaggle.com/code/jaykumar1607/brain-tumor-mri-classification-tensorflow-cnn中的MRI肿瘤检测，该实现基于EfficientNetB0模型，将使用ImageNet数据集的权重。
