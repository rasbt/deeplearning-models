![Python 3.7](https://img.shields.io/badge/Python-3.7-blue.svg)

# Deep Learning Models

A collection of various deep learning architectures, models, and tips for TensorFlow and PyTorch in Jupyter Notebooks.

## Traditional Machine Learning

- Perceptron [[TensorFlow 1](tensorflow1_ipynb/basic-ml/perceptron.ipynb)] [[PyTorch](pytorch_ipynb/basic-ml/perceptron.ipynb)]
- Logistic Regression [[TensorFlow 1](tensorflow1_ipynb/basic-ml/logistic-regression.ipynb)] [[PyTorch](pytorch_ipynb/basic-ml/logistic-regression.ipynb)]
- Softmax Regression (Multinomial Logistic Regression) [[TensorFlow 1](tensorflow1_ipynb/basic-ml/softmax-regression.ipynb)] [[PyTorch](pytorch_ipynb/basic-ml/softmax-regression.ipynb)]

## Multilayer Perceptrons

- Multilayer Perceptron [[TensorFlow 1](tensorflow1_ipynb/mlp/mlp-basic.ipynb)] [[PyTorch](pytorch_ipynb/mlp/mlp-basic.ipynb)]
- Multilayer Perceptron with Dropout [[TensorFlow 1](tensorflow1_ipynb/mlp/mlp-dropout.ipynb)] [[PyTorch](pytorch_ipynb/mlp/mlp-dropout.ipynb)]
- Multilayer Perceptron with Batch Normalization [[TensorFlow 1](tensorflow1_ipynb/mlp/mlp-batchnorm.ipynb)] [[PyTorch](pytorch_ipynb/mlp/mlp-batchnorm.ipynb)]
- Multilayer Perceptron with Backpropagation from Scratch [[TensorFlow 1](tensorflow1_ipynb/mlp/mlp-lowlevel.ipynb)] [[PyTorch](pytorch_ipynb/mlp/mlp-fromscratch__sigmoid-mse.ipynb)]


## Convolutional Neural Networks


#### Basic

- Convolutional Neural Network [[TensorFlow 1](tensorflow1_ipynb/cnn/cnn-basic.ipynb)] [[PyTorch](pytorch_ipynb/cnn/cnn-basic.ipynb)]
- Convolutional Neural Network with He Initialization  [[PyTorch](pytorch_ipynb/cnn/cnn-he-init.ipynb)]

#### Concepts

- Replacing Fully-Connnected by Equivalent Convolutional Layers [[PyTorch](pytorch_ipynb/cnn/fc-to-conv.ipynb)]


#### Fully Convolutional

- Fully Convolutional Neural Network [[PyTorch](pytorch_ipynb/cnn/cnn-allconv.ipynb)]

#### AlexNet

- AlexNet on CIFAR-10 [[PyTorch](pytorch_ipynb/cnn/cnn-alexnet-cifar10.ipynb)]

#### VGG

- Convolutional Neural Network VGG-16 [[TensorFlow 1](tensorflow1_ipynb/cnn/cnn-vgg16.ipynb)] [[PyTorch](pytorch_ipynb/cnn/cnn-vgg16.ipynb)]
- VGG-16 Gender Classifier Trained on CelebA [[PyTorch](pytorch_ipynb/cnn/cnn-vgg16-celeba.ipynb)]
- Convolutional Neural Network VGG-19 [[PyTorch](pytorch_ipynb/cnn/cnn-vgg19.ipynb)]

#### ResNet

- ResNet and Residual Blocks [[PyTorch](pytorch_ipynb/cnn/resnet-ex-1.ipynb)]
- ResNet-18 Digit Classifier Trained on MNIST [[PyTorch](pytorch_ipynb/cnn/cnn-resnet18-mnist.ipynb)]
- ResNet-18 Gender Classifier Trained on CelebA [[PyTorch](pytorch_ipynb/cnn/cnn-resnet18-celeba-dataparallel.ipynb)]
- ResNet-34 Digit Classifier Trained on MNIST [[PyTorch](pytorch_ipynb/cnn/cnn-resnet34-mnist.ipynb)]
- ResNet-34 Gender Classifier Trained on CelebA [[PyTorch](pytorch_ipynb/cnn/cnn-resnet34-celeba-dataparallel.ipynb)]
- ResNet-50 Digit Classifier Trained on MNIST [[PyTorch](pytorch_ipynb/cnn/cnn-resnet50-mnist.ipynb)]
- ResNet-50 Gender Classifier Trained on CelebA [[PyTorch](pytorch_ipynb/cnn/cnn-resnet50-celeba-dataparallel.ipynb)]
- ResNet-101 Gender Classifier Trained on CelebA [[PyTorch](pytorch_ipynb/cnn/cnn-resnet101-celeba.ipynb)]
- ResNet-101 Trained on CIFAR-10 [[PyTorch](pytorch_ipynb/cnn/cnn-resnet101-cifar10.ipynb)]
- ResNet-152 Gender Classifier Trained on CelebA [[PyTorch](pytorch_ipynb/cnn/cnn-resnet152-celeba.ipynb)]

#### Network in Network

- Network in Network CIFAR-10 Classifier [[PyTorch](pytorch_ipynb/cnn/nin-cifar10.ipynb)] 


## Metric Learning

- Siamese Network with Multilayer Perceptrons [[TensorFlow 1](tensorflow1_ipynb/metric/siamese-1.ipynb)]

## Autoencoders

#### Fully-connected Autoencoders

- Autoencoder [[TensorFlow 1](tensorflow1_ipynb/autoencoder/ae-basic.ipynb)] [[PyTorch](pytorch_ipynb/autoencoder/ae-basic.ipynb)]

#### Convolutional Autoencoders

- Convolutional Autoencoder with Deconvolutions / Transposed Convolutions[[TensorFlow 1](tensorflow1_ipynb/autoencoder/ae-deconv.ipynb)] [[PyTorch](pytorch_ipynb/autoencoder/ae-deconv.ipynb)]
- Convolutional Autoencoder with Deconvolutions (without pooling operations) [[PyTorch](pytorch_ipynb/autoencoder/ae-deconv-nopool.ipynb)]
- Convolutional Autoencoder with Nearest-neighbor Interpolation [[TensorFlow 1](tensorflow1_ipynb/autoencoder/ae-conv-nneighbor.ipynb)] [[PyTorch](pytorch_ipynb/autoencoder/ae-conv-nneighbor.ipynb)]
- Convolutional Autoencoder with Nearest-neighbor Interpolation -- Trained on CelebA [[PyTorch](pytorch_ipynb/autoencoder/ae-conv-nneighbor-celeba.ipynb)]
- Convolutional Autoencoder with Nearest-neighbor Interpolation -- Trained on Quickdraw [[PyTorch](pytorch_ipynb/autoencoder/ae-conv-nneighbor-quickdraw-1.ipynb)]

#### Variational Autoencoders

- Variational Autoencoder [[PyTorch](pytorch_ipynb/autoencoder/ae-var.ipynb)]
- Convolutional Variational Autoencoder [[PyTorch](pytorch_ipynb/autoencoder/ae-conv-var.ipynb)]

#### Conditional Variational Autoencoders

- Conditional Variational Autoencoder (with labels in reconstruction loss) [[PyTorch](pytorch_ipynb/autoencoder/ae-cvae.ipynb)]
- Conditional Variational Autoencoder (without labels in reconstruction loss) [[PyTorch](pytorch_ipynb/autoencoder/ae-cvae_no-out-concat.ipynb)]
- Convolutional Conditional Variational Autoencoder (with labels in reconstruction loss) [[PyTorch](pytorch_ipynb/autoencoder/ae-cnn-cvae.ipynb)]
- Convolutional Conditional Variational Autoencoder (without labels in reconstruction loss) [[PyTorch](pytorch_ipynb/autoencoder/ae-cnn-cvae_no-out-concat.ipynb)]

## Generative Adversarial Networks (GANs)

- Fully Connected GAN on MNIST [[TensorFlow 1](tensorflow1_ipynb/gan/gan.ipynb)] [[PyTorch](pytorch_ipynb/gan/gan.ipynb)]
- Convolutional GAN on MNIST [[TensorFlow 1](tensorflow1_ipynb/gan/gan-conv.ipynb)] [[PyTorch](pytorch_ipynb/gan/gan-conv.ipynb)]
- Convolutional GAN on MNIST with Label Smoothing [[TensorFlow 1](tensorflow1_ipynb/gan/gan-conv-smoothing.ipynb)] [[PyTorch](pytorch_ipynb/gan/gan-conv-smoothing.ipynb)]

## Recurrent Neural Networks (RNNs)


#### Many-to-one: Sentiment Analysis / Classification

- A simple single-layer RNN (IMDB) [[PyTorch](pytorch_ipynb/rnn/rnn_simple_imdb.ipynb)]
- A simple single-layer RNN with packed sequences to ignore padding characters (IMDB) [[PyTorch](pytorch_ipynb/rnn/rnn_simple_packed_imdb.ipynb)]
- RNN with LSTM cells (IMDB) [[PyTorch](pytorch_ipynb/rnn/rnn_lstm_packed_imdb.ipynb)]
- RNN with LSTM cells (IMDB) and pre-trained GloVe word vectors [[PyTorch](pytorch_ipynb/rnn/rnn_lstm_packed_imdb-glove.ipynb)]
- RNN with LSTM cells and Own Dataset in CSV Format (IMDB) [[PyTorch](pytorch_ipynb/rnn/rnn_lstm_packed_own_csv_imdb.ipynb)]
- RNN with GRU cells (IMDB) [[PyTorch](pytorch_ipynb/rnn/rnn_gru_packed_imdb.ipynb)]
- Multilayer bi-directional RNN (IMDB) [[PyTorch](pytorch_ipynb/rnn/rnn_gru_packed_imdb.ipynb)]

#### Many-to-Many / Sequence-to-Sequence

- A simple character RNN to generate new text (Charles Dickens) [[PyTorch](pytorch_ipynb/rnn/char_rnn-charlesdickens.ipynb)]



## Ordinal Regression

- Ordinal Regression CNN -- CORAL w. ResNet34 on AFAD-Lite [[PyTorch](pytorch_ipynb/ordinal/ordinal-cnn-coral-afadlite.ipynb)]
- Ordinal Regression CNN -- Niu et al. 2016 w. ResNet34 on AFAD-Lite [[PyTorch](pytorch_ipynb/ordinal/ordinal-cnn-niu-afadlite.ipynb)]
- Ordinal Regression CNN -- Beckham and Pal 2016 w. ResNet34 on AFAD-Lite [[PyTorch](pytorch_ipynb/ordinal/ordinal-cnn-beckham2016-afadlite.ipynb)]






## Tips and Tricks

- Cyclical Learning Rate [[PyTorch](pytorch_ipynb/tricks/cyclical-learning-rate.ipynb)]



## PyTorch Workflows and Mechanics

#### Custom Datasets

- Using PyTorch Dataset Loading Utilities for Custom Datasets -- CSV files converted to HDF5 [[PyTorch](pytorch_ipynb/mechanics/custom-data-loader-csv.ipynb)]
- Using PyTorch Dataset Loading Utilities for Custom Datasets -- Face Images from CelebA [[PyTorch](pytorch_ipynb/mechanics/custom-data-loader-celeba.ipynb)]
- Using PyTorch Dataset Loading Utilities for Custom Datasets -- Drawings from Quickdraw [[PyTorch](pytorch_ipynb/mechanics/custom-data-loader-quickdraw.ipynb)]
- Using PyTorch Dataset Loading Utilities for Custom Datasets -- Drawings from the Street View House Number (SVHN) Dataset [[PyTorch](pytorch_ipynb/mechanics/custom-data-loader-svhn.ipynb)]
- Using PyTorch Dataset Loading Utilities for Custom Datasets -- Asian Face Dataset (AFAD) [[PyTorch](pytorch_ipynb/mechanics/custom-data-loader-afad.ipynb)]
- Using PyTorch Dataset Loading Utilities for Custom Datasets -- Dating Historical Color Images [[PyTorch](pytorch_ipynb/mechanics/custom-data-loader_dating-historical-color-images.ipynb)]

#### Training and Preprocessing

- Generating Validation Set Splits [[PyTorch]](pytorch_ipynb/mechanics/validation-splits.ipynb)]
- Dataloading with Pinned Memory [[PyTorch](pytorch_ipynb/cnn/cnn-resnet34-cifar10-pinmem.ipynb)]
- Standardizing Images [[PyTorch](pytorch_ipynb/cnn/cnn-standardized.ipynb)]
- Image Transformation Examples [[PyTorch](pytorch_ipynb/mechanics/torchvision-transform-examples.ipynb)]
- Char-RNN with Own Text File [[PyTorch](pytorch_ipynb/rnn/char_rnn-charlesdickens.ipynb)]
- Sentiment Classification RNN with Own CSV File [[PyTorch](pytorch_ipynb/rnn/rnn_lstm_packed_own_csv_imdb.ipynb)]


#### Parallel Computing

- Using Multiple GPUs with DataParallel -- VGG-16 Gender Classifier on CelebA [[PyTorch](pytorch_ipynb/cnn/cnn-vgg16-celeba-data-parallel.ipynb)]

#### Other 

- Sequential API and hooks  [[PyTorch](pytorch_ipynb/mechanics/mlp-sequential.ipynb)]
- Weight Sharing Within a Layer  [[PyTorch](pytorch_ipynb/mechanics/cnn-weight-sharing.ipynb)]
- Plotting Live Training Performance in Jupyter Notebooks with just Matplotlib  [[PyTorch](pytorch_ipynb/mechanics/plot-jupyter-matplotlib.ipynb)]

#### Autograd

- Getting Gradients of an Intermediate Variable in PyTorch  [[PyTorch](pytorch_ipynb/mechanics/manual-gradients.ipynb)]



## TensorFlow Workflows and Mechanics

#### Custom Datasets

- Chunking an Image Dataset for Minibatch Training using NumPy NPZ Archives [[TensorFlow 1](tensorflow1_ipynb/mechanics/image-data-chunking-npz.ipynb)]
- Storing an Image Dataset for Minibatch Training using HDF5 [[TensorFlow 1](tensorflow1_ipynb/mechanics/image-data-chunking-hdf5.ipynb)]
- Using Input Pipelines to Read Data from TFRecords Files [[TensorFlow 1](tensorflow1_ipynb/mechanics/tfrecords.ipynb)]
- Using Queue Runners to Feed Images Directly from Disk [[TensorFlow 1](tensorflow1_ipynb/mechanics/file-queues.ipynb)]
- Using TensorFlow's Dataset API [[TensorFlow 1](tensorflow1_ipynb/mechanics/dataset-api.ipynb)]

#### Training and Preprocessing

- Saving and Loading Trained Models -- from TensorFlow Checkpoint Files and NumPy NPZ Archives [[TensorFlow 1](tensorflow1_ipynb/mechanics/saving-and-reloading-models.ipynb)]


