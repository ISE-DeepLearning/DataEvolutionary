# -*- coding=utf-8 -*-
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import os
import config
import scipy.misc

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# if not exists then just makedirs
if not os.path.exists(config.training_npy_save_path):
    os.makedirs(config.training_npy_save_path)
if not os.path.exists(config.training_pics_save_path):
    os.makedirs(config.training_pics_save_path)
if not os.path.exists(config.test_npy_save_path):
    os.makedirs(config.test_npy_save_path)
if not os.path.exists(config.test_pics_save_path):
    os.makedirs(config.test_pics_save_path)


def save_training_npy():
    images = mnist.train.images
    labels = mnist.train.labels
    images_classify_data = [[], [], [], [], [], [], [], [], [], []]
    # 按照0-9将测试数据分开
    for i in range(len(labels)):
        print(i)
        pos = list(labels[i]).index(1)
        images_classify_data[pos].append(images[i])

    for i in range(len(images_classify_data)):
        if not os.path.exists(config.training_npy_save_path):
            os.makedirs(config.training_npy_save_path)
        save_path = os.path.join(config.training_npy_save_path, str(i) + '.npy')
        np.save(save_path, images_classify_data[i])


def save_training_pics():
    images = mnist.train.images
    labels = mnist.train.labels
    for i in range(len(labels)):
        print(i)
        num = list(labels[i]).index(1)
        # # 灰度图
        save_dir = os.path.join(config.training_pics_save_path, str(num))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'train_' + str(num) + '_' + str(i) + '.png')
        scipy.misc.toimage(images[i].reshape((28, 28)), cmin=0.0, cmax=1.0).save(save_path)


def save_test_npy():
    images = mnist.test.images
    labels = mnist.test.labels
    images_classify_data = [[], [], [], [], [], [], [], [], [], []]
    # 按照0-9将测试数据分开
    for i in range(len(labels)):
        print(i)
        pos = list(labels[i]).index(1)
        images_classify_data[pos].append(images[i])

    for i in range(len(images_classify_data)):
        if not os.path.exists(config.training_npy_save_path):
            os.makedirs(config.training_npy_save_path)
        save_path = os.path.join(config.test_npy_save_path, str(i) + '.npy')
        np.save(save_path, images_classify_data[i])


def save_test_pics():
    images = mnist.test.images
    labels = mnist.test.labels
    for i in range(len(labels)):
        print(i)
        num = list(labels[i]).index(1)
        print(np.shape(images[i]))
        # # 灰度图
        save_dir = os.path.join(config.test_pics_save_path, str(num))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'test_' + str(num) + '_' + str(i) + '.png')
        scipy.misc.toimage(images[i].reshape((28, 28)), cmin=0.0, cmax=1.0).save(save_path)
        # image = Image.fromarray(np.reshape(images[i], (28, 28)),'L')
        # save_dir = os.path.join(config.test_pics_save_path, str(num))
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # save_path = os.path.join(save_dir, 'test_' + str(num) + '_' + str(i) + '.png')
        # image.save(save_path)


def test():
    images_npy_datas = np.load('../training_npy/0.npy')

    data = Image.open('../training_pics/0/train_0_7.png').convert('L')
    data = np.array(data).reshape((784,))
    print(images_npy_datas[0].reshape((28, 28)))
    print('[')
    for i in range(28):
        data = [str(images_npy_datas[0].reshape((28, 28))[i][j]) for j in range(28)]
        data_str = '[' + (','.join(data)) + '],'
        print(data_str)
    print(']')
    # 经过对比 0.npy的第一条数据 同 train_0的第一张图片二者的数据是相同的，验证数据的相同
    # print(data == (images_npy_datas[0] * 255).astype(np.uint8))
    # not recommanded 浮点数的精度问题决定了不适合进行比较
    # print(data/255 == images_npy_datas[0])


if __name__ == '__main__':
    # save_training_npy()
    # save_test_npy()
    # save_training_pics()
    # save_test_pics()
    test()
    pass
