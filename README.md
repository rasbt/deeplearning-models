![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)

# 深度学习模型

本项目讲述了深度学习中的结构、模型和技巧，使用的深度学习框架是 TensorFlow 和 PyTorch，代码和图文都以 Jupyter Notebook 的形式编写。

## 传统机器学习

- 感知器 [[TensorFlow 1](tensorflow1_ipynb/basic-ml/perceptron.ipynb)] [[PyTorch](pytorch_ipynb/basic-ml/perceptron.ipynb)]
- 逻辑回归（二分类器） [[TensorFlow 1](tensorflow1_ipynb/basic-ml/logistic-regression.ipynb)] [[PyTorch](pytorch_ipynb/basic-ml/logistic-regression.ipynb)]
- Softmax 回归（多分类器） [[TensorFlow 1](tensorflow1_ipynb/basic-ml/softmax-regression.ipynb)] [[PyTorch](pytorch_ipynb/basic-ml/softmax-regression.ipynb)]

## 多层感知器

- 多层感知器 [[TensorFlow 1](tensorflow1_ipynb/mlp/mlp-basic.ipynb)] [[PyTorch](pytorch_ipynb/mlp/mlp-basic.ipynb)]
- 带有 Dropout 的多层感知器 [[TensorFlow 1](tensorflow1_ipynb/mlp/mlp-dropout.ipynb)] [[PyTorch](pytorch_ipynb/mlp/mlp-dropout.ipynb)]
- 带有 Batch Normalization 的多层感知器 [[TensorFlow 1](tensorflow1_ipynb/mlp/mlp-batchnorm.ipynb)] [[PyTorch](pytorch_ipynb/mlp/mlp-batchnorm.ipynb)]
- 手写反向传播的多层感知器 [[TensorFlow 1](tensorflow1_ipynb/mlp/mlp-lowlevel.ipynb)] [[PyTorch](pytorch_ipynb/mlp/mlp-fromscratch__sigmoid-mse.ipynb)]

## 卷积神经网络

#### 基本

- 卷积神经网络 [[TensorFlow 1](tensorflow1_ipynb/cnn/cnn-basic.ipynb)] [[PyTorch](pytorch_ipynb/cnn/cnn-basic.ipynb)]
- 使用 He 初始化的卷积神经网络  [[PyTorch](pytorch_ipynb/cnn/cnn-he-init.ipynb)]

#### 概念

- 使用卷积层等效替换全连接层 [[PyTorch](pytorch_ipynb/cnn/fc-to-conv.ipynb)]

#### 全卷积

- 全卷积网络 [[PyTorch](pytorch_ipynb/cnn/cnn-allconv.ipynb)]

#### 数据集介绍

| 数据集 | 中文名称 | 样本数 | 图像尺寸 | 官方网站 |
| ---- | ---- | ---- | ---- | ---- |
| MNIST | 手写数字数据集 | 训练集 60000，测试集 10000 | (28, 28) | [MNIST](http://yann.lecun.com/exdb/mnist/) | 
| CIFAR-10 | 加拿大高等研究院-10 | 训练集 50000，测试集 10000 | (32, 32) | [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) | 
| SVHN | 街景门牌号 | 训练集 73257，测试集 26032，额外 531131 | 尺寸不一，裁剪后 (32, 32) | [SVHN](http://ufldl.stanford.edu/housenumbers/) |
| CelebA | 名人面部属性数据集 | 202599 | 尺寸不一，图像宽度超过 200 | [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| Quickdraw | 快速涂鸦数据集 | 5000 万 | 原始尺寸是 (256, 256)，裁剪后为 (32, 32) | [Quickdraw](https://github.com/googlecreativelab/quickdraw-dataset) |

#### 模型搭建与训练

| 数据集 | 模型 | 任务 | 地址 | 测试集准确率 |
| ---- | ---- | ---- | ---- | ---- |
| CIFAR-10 | AlexNet | 图像分类 | [PyTorch](pytorch_ipynb/cnn/cnn-alexnet-cifar10.ipynb) | 66.63% |
| CIFAR-10 | VGG-16 | 图像分类 | [PyTorch](pytorch_ipynb/cnn/cnn-vgg16.ipynb) | 75.43% |
| CIFAR-10 | VGG-19 | 图像分类 | [PyTorch](pytorch_ipynb/cnn/cnn-vgg19.ipynb) | 74.56% |
| CIFAR-10 | Network in Network | 图像分类 | [PyTorch](pytorch_ipynb/cnn/nin-cifar10.ipynb) | 70.67% |
| MNIST | ResNet 残差模块练习 | 数字分类 | [PyTorch](pytorch_ipynb/cnn/resnet-ex-1.ipynb) | 97.91% |
| MNIST | ResNet-18 | 数字分类 | [PyTorch](pytorch_ipynb/cnn/cnn-resnet18-mnist.ipynb) | 99.06% |
| MNIST | ResNet-34 | 数字分类 | [PyTorch](pytorch_ipynb/cnn/cnn-resnet34-mnist.ipynb) | 99.04% |
| MNIST | ResNet-50 | 数字分类 | [PyTorch](pytorch_ipynb/cnn/cnn-resnet50-mnist.ipynb) | 98.39% |
| CelebA | VGG-16 | 性别分类 | [PyTorch](pytorch_ipynb/cnn/cnn-vgg16-celeba.ipynb) | 90.72% |
| CelebA | ResNet-18 | 性别分类 | [PyTorch](pytorch_ipynb/cnn/cnn-resnet18-celeba-dataparallel.ipynb) | 97.38% |
| CelebA | ResNet-34 | 性别分类 | [PyTorch](pytorch_ipynb/cnn/cnn-resnet34-celeba-dataparallel.ipynb) | 97.56% |
| CelebA | ResNet-50 | 性别分类 | [PyTorch](pytorch_ipynb/cnn/cnn-resnet50-celeba-dataparallel.ipynb) | 97.40% |
| CelebA | ResNet-101 | 性别分类 | [PyTorch](pytorch_ipynb/cnn/cnn-resnet101-celeba.ipynb) | 97.52% |
| CelebA | ResNet-152 | 性别分类 | [PyTorch](pytorch_ipynb/cnn/cnn-resnet152-celeba.ipynb) |

## 度量学习

- 多层感知器实现的孪生网络 [[TensorFlow 1](tensorflow1_ipynb/metric/siamese-1.ipynb)]

## 自编码器

#### 全连接自编码器

- 自编码器 [[TensorFlow 1](tensorflow1_ipynb/autoencoder/ae-basic.ipynb)] [[PyTorch](pytorch_ipynb/autoencoder/ae-basic.ipynb)]

#### 卷积自编码器

- 反卷积 / 转置卷积实现的卷积自编码器[[TensorFlow 1](tensorflow1_ipynb/autoencoder/ae-deconv.ipynb)] [[PyTorch](pytorch_ipynb/autoencoder/ae-deconv.ipynb)]
- 转置卷积实现的卷积自编码器（没有使用池化操作） [[PyTorch](pytorch_ipynb/autoencoder/ae-deconv-nopool.ipynb)]
- 最近邻插值实现的卷积自编码器 [[TensorFlow 1](tensorflow1_ipynb/autoencoder/ae-conv-nneighbor.ipynb)] [[PyTorch](pytorch_ipynb/autoencoder/ae-conv-nneighbor.ipynb)]
- 在 CelebA 上训练的最近邻插值卷积自编码器 [[PyTorch](pytorch_ipynb/autoencoder/ae-conv-nneighbor-celeba.ipynb)]
- 在 Quickdraw 上训练的最近邻插值卷积自编码器 [[PyTorch](pytorch_ipynb/autoencoder/ae-conv-nneighbor-quickdraw-1.ipynb)]

#### 变分自动编码器

- 变分自动编码器 [[PyTorch](pytorch_ipynb/autoencoder/ae-var.ipynb)]
- 卷积变分自动编码器 [[PyTorch](pytorch_ipynb/autoencoder/ae-conv-var.ipynb)]

#### 条件变分自动编码器

- 条件变分自动编码器（重建损失中带标签） [[PyTorch](pytorch_ipynb/autoencoder/ae-cvae.ipynb)]
- 条件变分自动编码器（重建损失中没有标签） [[PyTorch](pytorch_ipynb/autoencoder/ae-cvae_no-out-concat.ipynb)]
- 卷积条件变分自动编码器（重建损失中带标签） [[PyTorch](pytorch_ipynb/autoencoder/ae-cnn-cvae.ipynb)]
- 卷积条件变分自动编码器（重建损失中没有标签） [[PyTorch](pytorch_ipynb/autoencoder/ae-cnn-cvae_no-out-concat.ipynb)]

## 生成对抗网络 (GANs)

- 在 MNIST 上训练的全连接 GAN [[TensorFlow 1](tensorflow1_ipynb/gan/gan.ipynb)] [[PyTorch](pytorch_ipynb/gan/gan.ipynb)]
- 在 MNIST 上训练的卷积 GAN [[TensorFlow 1](tensorflow1_ipynb/gan/gan-conv.ipynb)] [[PyTorch](pytorch_ipynb/gan/gan-conv.ipynb)]
- 在 MNIST 上使用标签平滑训练的卷积 GAN [[PyTorch](pytorch_ipynb/gan/gan-conv-smoothing.ipynb)]

## 递归神经网络 (RNNs)

#### 多对一：情感分析、分类

- 一个简单的单层RNN（IMDB）[[PyTorch](pytorch_ipynb/rnn/rnn_simple_imdb.ipynb)]
- 一个简单的单层RNN，带有打包序列，用于忽略填充字符（IMDB） [[PyTorch](pytorch_ipynb/rnn/rnn_simple_packed_imdb.ipynb)]
- 带有长短期记忆（LSTM）的RNN（IMDB） [[PyTorch](pytorch_ipynb/rnn/rnn_lstm_packed_imdb.ipynb)]
- 带有长短期记忆（LSTM）的RNN，使用预训练 GloVe 词向量 [[PyTorch](pytorch_ipynb/rnn/rnn_lstm_packed_imdb-glove.ipynb)]
- 带有长短期记忆（LSTM）的RNN，训练 CSV 格式的数据集（IMDB）[[PyTorch](pytorch_ipynb/rnn/rnn_lstm_packed_own_csv_imdb.ipynb)]
- 带有门控单元（GRU）的RNN（IMDB） [[PyTorch](pytorch_ipynb/rnn/rnn_gru_packed_imdb.ipynb)]
- 多层双向RNN（IMDB） [[PyTorch](pytorch_ipynb/rnn/rnn_gru_packed_imdb.ipynb)]

#### 多对多 / 序列对序列

- Char-RNN 实现的文本生成器（Charles Dickens） [[PyTorch](pytorch_ipynb/rnn/char_rnn-charlesdickens.ipynb)]

## 序数回归

- 序数回归 CNN -- CORAL w. ResNet34（AFAD-Lite） [[PyTorch](pytorch_ipynb/ordinal/ordinal-cnn-coral-afadlite.ipynb)]
- 序数回归 CNN -- Niu et al. 2016 w. ResNet34（AFAD-Lite） [[PyTorch](pytorch_ipynb/ordinal/ordinal-cnn-niu-afadlite.ipynb)]
- 序数回归 CNN -- Beckham and Pal 2016 w. ResNet34（AFAD-Lite） [[PyTorch](pytorch_ipynb/ordinal/ordinal-cnn-niu-afadlite.ipynb)]

## 技巧和窍门

- 循环学习率 [[PyTorch](pytorch_ipynb/tricks/cyclical-learning-rate.ipynb)]

## PyTorch 工作流程和机制

#### 自定义数据集

- 使用 torch.utils.data 加载自定义数据集 -- CSV 文件转换为 HDF5 格式 [[PyTorch](pytorch_ipynb/mechanics/custom-data-loader-csv.ipynb)]
- 使用 torch.utils.data 加载自定义数据集 -- 来自 CelebA 的面部图像 [[PyTorch](pytorch_ipynb/mechanics/custom-data-loader-celeba.ipynb)]
- 使用 torch.utils.data 加载自定义数据集 -- 来自 Quickdraw 的手绘图像 [[PyTorch](pytorch_ipynb/mechanics/custom-data-loader-quickdraw.ipynb)]
- 使用 torch.utils.data 加载自定义数据集 -- 来自街景门牌号数据集（SVHN）的图像 [[PyTorch](pytorch_ipynb/mechanics/custom-data-loader-svhn.ipynb)]

#### 训练和预处理

- 在 DataLoader 中使用固定内存（pin_memory）技术 [[PyTorch](pytorch_ipynb/cnn/cnn-resnet34-cifar10-pinmem.ipynb)]
- 标准化图像（Standardization） [[PyTorch](pytorch_ipynb/cnn/cnn-standardized.ipynb)]
- 使用 torchvision 进行图像变换（数据增强） [[PyTorch](pytorch_ipynb/mechanics/torchvision-transform-examples.ipynb)]
- 在自己的文本数据上训练 Char-RNN [[PyTorch](pytorch_ipynb/rnn/char_rnn-charlesdickens.ipynb)]
- 在自己的文本数据集上使用 LSTM 进行情感分类 [[PyTorch](pytorch_ipynb/rnn/rnn_lstm_packed_own_csv_imdb.ipynb)]

#### 并行计算

- 使用 DataParallel 进行多 GPU 训练 -- 在 CelebA 上使用 VGG-16 训练性别分类器 [[PyTorch](pytorch_ipynb/cnn/cnn-vgg16-celeba-data-parallel.ipynb)]

#### 其他

- Sequential API 和 Hook 技术  [[PyTorch](pytorch_ipynb/mechanics/mlp-sequential.ipynb)]
- 同层权值共享  [[PyTorch](pytorch_ipynb/mechanics/cnn-weight-sharing.ipynb)]
- 使用 Matplotlib 在 Jupyter Notebook 中绘制实时训练曲线 [[PyTorch](pytorch_ipynb/mechanics/plot-jupyter-matplotlib.ipynb)]

#### Autograd

- 在 PyTorch 中获取中间变量的梯度 [[PyTorch](pytorch_ipynb/mechanics/manual-gradients.ipynb)]

## TensorFlow 工作流程和机制

#### 自定义数据集

- 使用 NumPy npz 格式打包小批量图像数据集 [[TensorFlow 1](tensorflow1_ipynb/mechanics/image-data-chunking-npz.ipynb)]
- 使用 HDF5 格式保存小批量图像数据集 [[TensorFlow 1](tensorflow1_ipynb/mechanics/image-data-chunking-hdf5.ipynb)]
- 使用输入管道在 TFRecords 文件中读取数据 [[TensorFlow 1](tensorflow1_ipynb/mechanics/tfrecords.ipynb)]
- 使用队列运行器（Queue Runners）从硬盘中直接读取图像 [[TensorFlow 1](tensorflow1_ipynb/mechanics/file-queues.ipynb)]
- 使用 TensorFlow 数据集 API [[TensorFlow 1](tensorflow1_ipynb/mechanics/dataset-api.ipynb)]

#### 训练和预处理

- 保存和加载模型 -- 保存为 TensorFlow Checkpoint 文件和 NumPy npz 文件 [[TensorFlow 1](tensorflow1_ipynb/mechanics/saving-and-reloading-models.ipynb)]
