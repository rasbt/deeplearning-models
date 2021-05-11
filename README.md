![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)

# Deep Learning Models

A collection of various deep learning architectures, models, and tips for TensorFlow and PyTorch in Jupyter Notebooks.

## Traditional Machine Learning

- Perceptron   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/basic-ml/perceptron.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/basic-ml/perceptron.ipynb)]   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/basic-ml/perceptron.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/basic-ml/perceptron.ipynb)]
- Logistic Regression  
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/basic-ml/logistic-regression.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/basic-ml/logistic-regression.ipynb)]   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/basic-ml/logistic-regression.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/basic-ml/logistic-regression.ipynb)]
- Softmax Regression (Multinomial Logistic Regression)  
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/basic-ml/softmax-regression.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/basic-ml/softmax-regression.ipynb)]   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/basic-ml/softmax-regression.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/basic-ml/softmax-regression.ipynb)]  
- Softmax Regression with MLxtend's plot_decision_regions on Iris  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/basic-ml/softmax-regression-mlxtend-1.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/basic-ml/softmax-regression-mlxtend-1.ipynb)]


## Multilayer Perceptrons

- Multilayer Perceptron   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/mlp/mlp-basic.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mlp/mlp-basic.ipynb)]  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mlp/mlp-basic.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-basic.ipynb)]
- Multilayer Perceptron with Dropout   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/mlp/mlp-dropout.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mlp/mlp-dropout.ipynb)]  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mlp/mlp-dropout.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-dropout.ipynb)]
- Multilayer Perceptron with Batch Normalization   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/mlp/mlp-batchnorm.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mlp/mlp-batchnorm.ipynb)]  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mlp/mlp-batchnorm.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-batchnorm.ipynb)]
- Multilayer Perceptron with Backpropagation from Scratch   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/mlp/mlp-lowlevel.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mlp/mlp-lowlevel.ipynb)]  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mlp/mlp-fromscratch__sigmoid-mse.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mlp/mlp-fromscratch__sigmoid-mse.ipynb)]


## Convolutional Neural Networks


#### Basic

- Convolutional Neural Network   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/cnn/cnn-basic.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/cnn/cnn-basic.ipynb)]  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-basic.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-basic.ipynb)]
- Convolutional Neural Network with He Initialization    
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-he-init.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-he-init.ipynb)]

#### Concepts

- Replacing Fully-Connnected by Equivalent Convolutional Layers   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/fc-to-conv.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/fc-to-conv.ipynb)]



---



#### AlexNet

- AlexNet on CIFAR-10   
  &nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-alexnet-cifar10.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-alexnet-cifar10.ipynb)]

#### DenseNet

- DenseNet-121 Digit Classifier Trained on MNIST   
  &nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-densenet121-mnist.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-densenet121-mnist.ipynb)]
- DenseNet-121 Image Classifier Trained on CIFAR-10   
  &nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-densenet121-cifar10.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-densenet121-cifar10.ipynb)]

#### Fully Convolutional

- Fully Convolutional Neural Network   
  &nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-allconv.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-allconv.ipynb)]

#### LeNet

- LeNet-5 on MNIST   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-lenet5-mnist.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-lenet5-mnist.ipynb)]
- LeNet-5 on CIFAR-10   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-lenet5-cifar10.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-lenet5-cifar10.ipynb)]  
- LeNet-5 on QuickDraw  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-lenet5-quickdraw.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-lenet5-quickdraw.ipynb)]



#### MobileNet

- MobileNet-v2 on Cifar-10  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-mobilenet-v2-cifar10.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-lenet5-quickdraw.ipynb)]
  
- MobileNet-v3 small on Cifar-10  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-mobilenet-v3-small-cifar10.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-lenet5-quickdraw.ipynb)]

- MobileNet-v3 large on Cifar-10  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-mobilenet-v3-large-cifar10.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-lenet5-quickdraw.ipynb)]



#### Network in Network

- Network in Network CIFAR-10 Classifier   
  &nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/nin-cifar10.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/nin-cifar10.ipynb)]  

#### VGG

- Convolutional Neural Network VGG-16 Trained on CIFAR-10  
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/cnn/cnn-vgg16.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/cnn/cnn-vgg16.ipynb)]  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-vgg16.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16.ipynb)]
- VGG-16 Gender Classifier Trained on CelebA   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-vgg16-celeba.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16-celeba.ipynb)]
- VGG-16 Dogs vs Cats Classifier  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-vgg16-cats-dogs.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16-cats-dogs.ipynb)]
- Convolutional Neural Network VGG-19   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-vgg19.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg19.ipynb)]

#### ResNet

- ResNet and Residual Blocks   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/resnet-ex-1.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/resnet-ex-1.ipynb)]
- ResNet-18 Digit Classifier Trained on MNIST   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet18-mnist.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet18-mnist.ipynb)]
- ResNet-18 Gender Classifier Trained on CelebA   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet18-celeba-dataparallel.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet18-celeba-dataparallel.ipynb)]
- ResNet-34 Digit Classifier Trained on MNIST   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet34-mnist.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet34-mnist.ipynb)]
- ResNet-34 Object Classifier Trained on QuickDraw  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet34-quickdraw.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet34-quickdraw.ipynb)]
- ResNet-34 Gender Classifier Trained on CelebA   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet34-celeba-dataparallel.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet34-celeba-dataparallel.ipynb)]
- ResNet-50 Digit Classifier Trained on MNIST   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet50-mnist.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet50-mnist.ipynb)]
- ResNet-50 Gender Classifier Trained on CelebA   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet50-celeba-dataparallel.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet50-celeba-dataparallel.ipynb)]
- ResNet-101 Gender Classifier Trained on CelebA   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet101-celeba.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet101-celeba.ipynb)]
- ResNet-101 Trained on CIFAR-10   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet101-cifar10.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet101-cifar10.ipynb)]
- ResNet-152 Gender Classifier Trained on CelebA   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet152-celeba.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet152-celeba.ipynb)]

---





## Normalization Layers

- BatchNorm before and after Activation for Network-in-Network CIFAR-10 Classifier     
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/nin-cifar10_batchnorm.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/nin-cifar10_batchnorm.ipynb)]  
- Filter Response Normalization for Network-in-Network CIFAR-10 Classifier  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/nin-cifar10_filter-response-norm.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/nin-cifar10_filter-response-norm.ipynb)] 



## Metric Learning

- Siamese Network with Multilayer Perceptrons   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/metric/siamese-1.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/metric/siamese-1.ipynb)]

## Autoencoders

#### Fully-connected Autoencoders

- Autoencoder (MNIST)  
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/autoencoder/ae-basic.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/autoencoder/ae-basic.ipynb)]  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-basic.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-basic.ipynb)]
- Autoencoder (MNIST) + Scikit-Learn Random Forest Classifier  
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/autoencoder/ae-basic-with-rf.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/autoencoder/ae-basic.ipynb)]  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-basic-with-rf.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-basic.ipynb)]



#### Convolutional Autoencoders

- Convolutional Autoencoder with Deconvolutions / Transposed Convolutions  
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/autoencoder/ae-deconv.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/autoencoder/ae-deconv.ipynb)]  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-deconv.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-deconv.ipynb)]
- Convolutional Autoencoder with Deconvolutions and Continuous Jaccard Distance  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-deconv-jaccard.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-deconv-jaccard.ipynb)]
- Convolutional Autoencoder with Deconvolutions (without pooling operations)   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-deconv-nopool.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-deconv-nopool.ipynb)]
- Convolutional Autoencoder with Nearest-neighbor Interpolation   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/autoencoder/ae-conv-nneighbor.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/autoencoder/ae-conv-nneighbor.ipynb)]  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-conv-nneighbor.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-conv-nneighbor.ipynb)]
- Convolutional Autoencoder with Nearest-neighbor Interpolation -- Trained on CelebA   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-conv-nneighbor-celeba.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-conv-nneighbor-celeba.ipynb)]
- Convolutional Autoencoder with Nearest-neighbor Interpolation -- Trained on Quickdraw   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-conv-nneighbor-quickdraw-1.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-conv-nneighbor-quickdraw-1.ipynb)]

#### Variational Autoencoders

- Variational Autoencoder   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-var.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-var.ipynb)]
- Convolutional Variational Autoencoder   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-conv-var.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-conv-var.ipynb)]

#### Conditional Variational Autoencoders

- Conditional Variational Autoencoder (with labels in reconstruction loss)   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-cvae.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-cvae.ipynb)]
- Conditional Variational Autoencoder (without labels in reconstruction loss)   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-cvae_no-out-concat.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-cvae_no-out-concat.ipynb)]
- Convolutional Conditional Variational Autoencoder (with labels in reconstruction loss)   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-cnn-cvae.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-cnn-cvae.ipynb)]
- Convolutional Conditional Variational Autoencoder (without labels in reconstruction loss)   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/autoencoder/ae-cnn-cvae_no-out-concat.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/autoencoder/ae-cnn-cvae_no-out-concat.ipynb)]

## Generative Adversarial Networks (GANs)

- Fully Connected GAN on MNIST   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/gan/gan.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/gan/gan.ipynb)]  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/gan/gan.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gan/gan.ipynb)]
- Fully Connected Wasserstein GAN on MNIST   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/gan/wgan-1.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gan/wgan-1.ipynb)]
- Convolutional GAN on MNIST   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/gan/gan-conv.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/gan/gan-conv.ipynb)]  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/gan/gan-conv.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gan/gan-conv.ipynb)]
- Convolutional GAN on MNIST with Label Smoothing   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/gan/gan-conv-smoothing.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/gan/gan-conv-smoothing.ipynb)]  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/gan/gan-conv-smoothing.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gan/gan-conv-smoothing.ipynb)]
- Convolutional Wasserstein GAN on MNIST  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/gan/dc-wgan-1.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gan/dc-wgan-1.ipynb)]
- "Deep Convolutional GAN" (DCGAN) on Cats and Dogs Images  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/gan/dcgan-cats-and-dogs.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gan/dcgan-cats-and-dogs.ipynb)]
- "Deep Convolutional GAN" (DCGAN) on CelebA Face Images  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/gan/dcgan-celeba.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gan/dcgan-celeba.ipynb)]

## Graph Neural Networks (GNNs)

- Most Basic Graph Neural Network with Gaussian Filter on MNIST    
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/gnn/gnn-basic-1.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gnn/gnn-basic-1.ipynb)]
- Basic Graph Neural Network with Edge Prediction on MNIST    
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/gnn/gnn-basic-edge-1.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gnn/gnn-basic-edge-1.ipynb)]
- Basic Graph Neural Network with Spectral Graph Convolution on MNIST  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/gnn/gnn-basic-graph-spectral-1.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/gnn/gnn-basic-graph-spectral-1.ipynb)]

## Recurrent Neural Networks (RNNs)


#### Many-to-one: Sentiment Analysis / Classification

- A simple single-layer RNN (IMDB)   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_simple_imdb.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_simple_imdb.ipynb)]
- A simple single-layer RNN with packed sequences to ignore padding characters (IMDB)   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_simple_packed_imdb.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_simple_packed_imdb.ipynb)]
- RNN with LSTM cells (IMDB)   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_lstm_packed_imdb.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_lstm_packed_imdb.ipynb)]
- RNN with LSTM cells (IMDB) and pre-trained GloVe word vectors   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_lstm_packed_imdb-glove.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_lstm_packed_imdb-glove.ipynb)]
- RNN with LSTM cells and Own Dataset in CSV Format (IMDB)   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_lstm_packed_own_csv_imdb.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_lstm_packed_own_csv_imdb.ipynb)]
- RNN with GRU cells (IMDB)   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_gru_packed_imdb.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_gru_packed_imdb.ipynb)]
- Multilayer bi-directional RNN (IMDB)   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_lstm_bi_imdb.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_lstm_bi_imdb.ipynb)]
- Bidirectional Multi-layer RNN with LSTM with Own Dataset in CSV Format (AG News)     
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_bi_multilayer_lstm_own_csv_agnews.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_bi_multilayer_lstm_own_csv_agnews.ipynb)]


#### Many-to-Many / Sequence-to-Sequence

- A simple character RNN to generate new text (Charles Dickens)   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/rnn/char_rnn-charlesdickens.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/char_rnn-charlesdickens.ipynb)]



## Ordinal Regression

- Ordinal Regression CNN -- CORAL w. ResNet34 on AFAD-Lite   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/ordinal/ordinal-cnn-coral-afadlite.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/ordinal/ordinal-cnn-coral-afadlite.ipynb)]
- Ordinal Regression CNN -- Niu et al. 2016 w. ResNet34 on AFAD-Lite   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/ordinal/ordinal-cnn-niu-afadlite.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/ordinal/ordinal-cnn-niu-afadlite.ipynb)]
- Ordinal Regression CNN -- Beckham and Pal 2016 w. ResNet34 on AFAD-Lite   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/ordinal/ordinal-cnn-beckham2016-afadlite.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/ordinal/ordinal-cnn-beckham2016-afadlite.ipynb)]





## Tips and Tricks

- Cyclical Learning Rate   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/tricks/cyclical-learning-rate.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/tricks/cyclical-learning-rate.ipynb)]
- Annealing with Increasing the Batch Size (w. CIFAR-10 & AlexNet)   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/tricks/cnn-alexnet-cifar10-batchincrease.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/tricks/cnn-alexnet-cifar10-batchincrease.ipynb)]
- Gradient Clipping (w. MLP on MNIST)   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/tricks/gradclipping_mlp.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/tricks/gradclipping_mlp.ipynb)]


## Transfer Learning

- Transfer Learning Example (VGG16 pre-trained on ImageNet for Cifar-10)  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/transfer/transferlearning-vgg16-cifar10-1.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/transfer/transferlearning-vgg16-cifar10-1.ipynb)]

## Visualization and Interpretation

- Vanilla Loss Gradient (wrt Inputs) Visualization (Based on a VGG16 Convolutional Neural Network for Kaggle's Cats and Dogs Images)  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/viz/cnns/cats-and-dogs/cnn-viz-grad__vgg16-cats-dogs.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/viz/cnns/cats-and-dogs/cnn-viz-grad__vgg16-cats-dogs.ipynb)]
- Guided Backpropagation (Based on a VGG16 Convolutional Neural Network for Kaggle's Cats and Dogs Images)  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/viz/cnns/cats-and-dogs/cnn-viz-guided-backprop__vgg16-cats-dogs.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/viz/cnns/cats-and-dogs/cnn-viz-guided-backprop__vgg16-cats-dogs.ipynb)]



## PyTorch Workflows and Mechanics

#### Custom Datasets

- Custom Data Loader Example for PNG Files  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/custom-dataloader-png/custom-dataloader-example.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-dataloader-png/custom-dataloader-example.ipynb)]
- Using PyTorch Dataset Loading Utilities for Custom Datasets -- CSV files converted to HDF5   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/custom-data-loader-csv.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader-csv.ipynb)]
- Using PyTorch Dataset Loading Utilities for Custom Datasets -- Face Images from CelebA   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/custom-data-loader-celeba.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader-celeba.ipynb)]
- Using PyTorch Dataset Loading Utilities for Custom Datasets -- Drawings from Quickdraw   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/custom-data-loader-quickdraw.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader-quickdraw.ipynb)]
- Using PyTorch Dataset Loading Utilities for Custom Datasets -- Drawings from the Street View House Number (SVHN) Dataset   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/custom-data-loader-svhn.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader-svhn.ipynb)]
- Using PyTorch Dataset Loading Utilities for Custom Datasets -- Asian Face Dataset (AFAD)   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/custom-data-loader-afad.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader-afad.ipynb)]
- Using PyTorch Dataset Loading Utilities for Custom Datasets -- Dating Historical Color Images   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/custom-data-loader_dating-historical-color-images.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader_dating-historical-color-images.ipynb)]
- Using PyTorch Dataset Loading Utilities for Custom Datasets -- Fashion MNIST   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/custom-data-loader-quickdraw.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/custom-data-loader-quickdraw.ipynb)]

#### Training and Preprocessing

- Generating Validation Set Splits   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/validation-splits.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/validation-splits.ipynb)]
- Dataloading with Pinned Memory   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-resnet34-cifar10-pinmem.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet34-cifar10-pinmem.ipynb)]
- Standardizing Images   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-standardized.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-standardized.ipynb)]
- Image Transformation Examples   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/torchvision-transform-examples.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/torchvision-transform-examples.ipynb)]
- Char-RNN with Own Text File   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/rnn/char_rnn-charlesdickens.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/char_rnn-charlesdickens.ipynb)]
- Sentiment Classification RNN with Own CSV File   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/rnn/rnn_lstm_packed_own_csv_imdb.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/rnn/rnn_lstm_packed_own_csv_imdb.ipynb)]


#### Improving Memory Efficiency

- Gradient Checkpointing Demo (Network-in-Network trained on CIFAR-10)  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/gradient-checkpointing-nin.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/gradient-checkpointing-nin.ipynb)]


#### Parallel Computing

- Using Multiple GPUs with DataParallel -- VGG-16 Gender Classifier on CelebA   
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/cnn/cnn-vgg16-celeba-data-parallel.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg16-celeba-data-parallel.ipynb)]
- Distribute a Model Across Multiple GPUs with Pipeline Parallelism (VGG-16 Example) 
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/model-pipeline-vgg16.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/model-pipeline-vgg16.ipynb)]

#### Other 

- PyTorch with and without Deterministic Behavior -- Runtime Benchmark  
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/deterministic_benchmark.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/pytorch_ipynb/mechanics/deterministic_benchmark.ipynb)]
- Sequential API and hooks    
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/mlp-sequential.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/mlp-sequential.ipynb)]
- Weight Sharing Within a Layer    
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/cnn-weight-sharing.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/cnn-weight-sharing.ipynb)]
- Plotting Live Training Performance in Jupyter Notebooks with just Matplotlib    
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/plot-jupyter-matplotlib.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/plot-jupyter-matplotlib.ipynb)]

#### Autograd

- Getting Gradients of an Intermediate Variable in PyTorch    
&nbsp;&nbsp; [PyTorch: [GitHub](pytorch_ipynb/mechanics/manual-gradients.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/manual-gradients.ipynb)]



## TensorFlow Workflows and Mechanics

#### Custom Datasets

- Chunking an Image Dataset for Minibatch Training using NumPy NPZ Archives   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/mechanics/image-data-chunking-npz.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/image-data-chunking-npz.ipynb)]
- Storing an Image Dataset for Minibatch Training using HDF5   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/mechanics/image-data-chunking-hdf5.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/image-data-chunking-hdf5.ipynb)]
- Using Input Pipelines to Read Data from TFRecords Files   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/mechanics/tfrecords.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/tfrecords.ipynb)]
- Using Queue Runners to Feed Images Directly from Disk   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/mechanics/file-queues.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/file-queues.ipynb)]
- Using TensorFlow's Dataset API   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/mechanics/dataset-api.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/dataset-api.ipynb)]

#### Training and Preprocessing

- Saving and Loading Trained Models -- from TensorFlow Checkpoint Files and NumPy NPZ Archives   
&nbsp;&nbsp; [TensorFlow 1: [GitHub](tensorflow1_ipynb/mechanics/saving-and-reloading-models.ipynb) | [Nbviewer](https://nbviewer.jupyter.org/github/rasbt/deeplearning-models/blob/master/tensorflow1_ipynb/mechanics/saving-and-reloading-models.ipynb)]

