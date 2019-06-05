# Sebastian Raschka 2016-2017
#
# Supporting code for the book
# "Introduction to Artificial Neural Networks and Deep Learning:
#  A Practical Guide with Applications in Python"
#
# Source: https://github.com/rasbt/deep-learning-book
# Author: Sebastian Raschka <sebastianraschka.com>
# License: MIT


from urllib.request import urlretrieve
import shutil
import glob
import tarfile
import os
import sys
import pickle
import numpy as np
import scipy.misc
from tensorflow.examples.tutorials.mnist import input_data


def download_and_extract_cifar(target_dir,
                               cifar_url='http://www.cs.toronto.edu/'
                               '~kriz/cifar-10-python.tar.gz'):

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    fbase = os.path.basename(cifar_url)
    fpath = os.path.join(target_dir, fbase)

    if not os.path.exists(fpath):
        def get_progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading ... %s %d%%' % (fbase,
                             float(count * block_size) /
                             float(total_size) * 100.0))
            sys.stdout.flush()
        local_filename, headers = urlretrieve(cifar_url,
                                              fpath,
                                              reporthook=get_progress)
        sys.stdout.write('\nDownloaded')

    else:
        sys.stdout.write('Found existing')

    statinfo = os.stat(fpath)
    file_size = statinfo.st_size / 1024**2
    sys.stdout.write(' %s (%.1f Mb)\n' % (fbase, file_size))
    sys.stdout.write('Extracting %s ...\n' % fbase)
    sys.stdout.flush()

    with tarfile.open(fpath, 'r:gz') as t:
        t.extractall(target_dir)

    return fpath.replace('cifar-10-python.tar.gz', 'cifar-10-batches-py')


def unpickle_cifar(fpath):
    with open(fpath, 'rb') as f:
        dct = pickle.load(f, encoding='bytes')
    return dct


class Cifar10Loader():
    def __init__(self, cifar_path, normalize=False,
                 channel_mean_center=False, zero_center=False):
        self.cifar_path = cifar_path
        self.batchnames = [os.path.join(self.cifar_path, f)
                           for f in os.listdir(self.cifar_path)
                           if f.startswith('data_batch')]
        self.testname = os.path.join(self.cifar_path, 'test_batch')
        self.num_train = self.count_train()
        self.num_test = self.count_test()
        self.normalize = normalize
        self.channel_mean_center = channel_mean_center
        self.zero_center = zero_center
        self.train_mean = None

    def _compute_train_mean(self):

        cum_mean = np.zeros((1, 1, 1, 3))

        for batch in self.batchnames:
            dct = unpickle_cifar(batch)
            dct[b'labels'] = np.array(dct[b'labels'], dtype=int)
            dct[b'data'] = dct[b'data'].reshape(
                dct[b'data'].shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
            mean = dct[b'data'].mean(axis=(0, 1, 2), keepdims=True)
            cum_mean += mean

        self.train_mean = cum_mean / len(self.batchnames)

        return None

    def load_test(self, onehot=True):
        dct = unpickle_cifar(self.testname)
        dct[b'labels'] = np.array(dct[b'labels'], dtype=int)

        dct[b'data'] = dct[b'data'].reshape(
            dct[b'data'].shape[0], 3, 32, 32).transpose(0, 2, 3, 1)

        if onehot:
            dct[b'labels'] = (np.arange(10) ==
                              dct[b'labels'][:, None]).astype(int)

        if self.normalize:
            dct[b'data'] = dct[b'data'].astype(np.float32)
            dct[b'data'] = dct[b'data'] / 255.0

        if self.channel_mean_center:
            if self.train_mean is None:
                self._compute_train_mean()
            dct[b'data'] -= self.train_mean

        if self.zero_center:
            if self.normalize:
                dct[b'data'] -= .5
            else:
                dct[b'data'] -= 127.5

        return dct[b'data'], dct[b'labels']

    def load_train_epoch(self, batch_size=50, onehot=True,
                         shuffle=False, seed=None):

        rgen = np.random.RandomState(seed)

        for batch in self.batchnames:
            dct = unpickle_cifar(batch)
            dct[b'labels'] = np.array(dct[b'labels'], dtype=int)
            dct[b'data'] = dct[b'data'].reshape(
                dct[b'data'].shape[0], 3, 32, 32).transpose(0, 2, 3, 1)

            if onehot:
                dct[b'labels'] = (np.arange(10) ==
                                  dct[b'labels'][:, None]).astype(int)

            if self.normalize:
                dct[b'data'] = dct[b'data'].astype(np.float32)
                dct[b'data'] = dct[b'data'] / 255.0

            if self.channel_mean_center:
                if self.train_mean is None:
                    self._compute_train_mean()
                dct[b'data'] -= self.train_mean

            if self.zero_center:
                if self.normalize:
                    dct[b'data'] -= .5
                else:
                    dct[b'data'] -= 127.5

            arrays = [dct[b'data'], dct[b'labels']]
            del dct
            indices = np.arange(arrays[0].shape[0])

            if shuffle:
                rgen.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - batch_size + 1,
                                   batch_size):
                index_slice = indices[start_idx:start_idx + batch_size]
                yield (ary[index_slice] for ary in arrays)

    def count_train(self):
        cnt = 0
        for f in self.batchnames:
            dct = unpickle_cifar(f)
            cnt += len(dct[b'labels'])
        return cnt

    def count_test(self):
        dct = unpickle_cifar(self.testname)
        return len(dct[b'labels'])


def mnist_export_to_jpg(path='./'):

    mnist = input_data.read_data_sets("./", one_hot=False)

    batch_x, batch_y = mnist.train.next_batch(50000)
    cnt = -1

    def remove_incomplete_existing(path_prefix, expect_files):
        dir_path = os.path.join(path, 'mnist_%s' % path_prefix)

        is_empty = False
        if not os.path.exists(dir_path):
            for i in range(10):
                outpath = os.path.join(path, dir_path, str(i))
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
            is_empty = True
        else:
            num_existing_files = len(glob.glob('%s/*/*.jpg' % dir_path))
            if num_existing_files > 0 and num_existing_files < expect_files:
                shutil.rmtree(dir_path)
                is_empty = True
                for i in range(10):
                    outpath = os.path.join(path, dir_path, str(i))
                    if not os.path.exists(outpath):
                        os.makedirs(outpath)
        return is_empty

    is_empty = remove_incomplete_existing(path_prefix='train',
                                          expect_files=45000)
    if is_empty:
        for data, label in zip(batch_x[:45000], batch_y[:45000]):
            cnt += 1
            outpath = os.path.join(path, 'mnist_train/%d/%05d.jpg' %
                                   (label, cnt))
            scipy.misc.imsave(outpath, (data*255).reshape(28, 28))

    is_empty = remove_incomplete_existing(path_prefix='valid',
                                          expect_files=5000)
    if is_empty:
        for data, label in zip(batch_x[45000:], batch_y[45000:]):
            cnt += 1
            outpath = os.path.join(path, 'mnist_valid/%d/%05d.jpg' %
                                   (label, cnt))
            scipy.misc.imsave(outpath, (data*255).reshape(28, 28))

    is_empty = remove_incomplete_existing(path_prefix='test',
                                          expect_files=10000)
    if is_empty:
        batch_x, batch_y = mnist.test.next_batch(10000)
        cnt = -1
        for data, label in zip(batch_x, batch_y):
            cnt += 1
            outpath = os.path.join(path, 'mnist_test/%d/%05d.jpg' % (label, cnt))
            scipy.misc.imsave(outpath, (data*255).reshape(28, 28))
