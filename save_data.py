# -*- coding=utf-8 -*-
import numpy as np
import pickle
import matplotlib.pyplot as plt
import config
import os
from tensorflow.examples.tutorials.mnist import input_data


def mkdir():
    if not os.path.exists(config.datasets['cifar10']['training_dir_path']):
        os.makedirs(config.datasets['cifar10']['training_dir_path'])
    if not os.path.exists(config.datasets['cifar10']['test_dir_path']):
        os.makedirs(config.datasets['cifar10']['test_dir_path'])
    if not os.path.exists(config.datasets['mnist']['training_dir_path']):
        os.makedirs(config.datasets['mnist']['training_dir_path'])
    if not os.path.exists(config.datasets['mnist']['test_dir_path']):
        os.makedirs(config.datasets['mnist']['training_dir_path'])


def save_cifar10_train():
    datasets = [[], [], [], [], [], [], [], [], [], []]
    for i in range(1, 6):
        address = os.path.join(config.datasets['cifar10']['origin_dir_path'], 'data_batch_' + str(i))
        with open(address, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        for j in range(10000):
            label = data['labels'][j]
            pic = data['data'][j]
            datasets[label].append(list(pic))
    for i in range(10):
        address = os.path.join(config.datasets['cifar10']['training_dir_path'], str(i) + '.npy')
        np.save(address, datasets[i])


def save_cifar10_test():
    datasets = [[], [], [], [], [], [], [], [], [], []]
    address = os.path.join(config.datasets['cifar10']['origin_dir_path'], 'test_batch')
    with open(address, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    for j in range(10000):
        label = data['labels'][j]
        pic = data['data'][j]
        datasets[label].append(list(pic))
        print(j)
    for i in range(10):
        address = os.path.join(config.datasets['cifar10']['test_dir_path'], str(i) + '.npy')
        np.save(address, datasets[i])


def save_mnist_training():
    mnist = input_data.read_data_sets(config.datasets['mnist']['origin_dir_path'], one_hot=True)
    images = mnist.train.images
    labels = mnist.train.labels
    images_classify_data = [[], [], [], [], [], [], [], [], [], []]
    for i in range(len(labels)):
        pos = list(labels[i]).index(1)
        images_classify_data[pos].append(images[i])

    for i in range(len(images_classify_data)):
        save_path = os.path.join(config.datasets['mnist']['training_dir_path'], str(i) + '.npy')
        np.save(save_path, images_classify_data[i])


def save_mnist_test():
    mnist = input_data.read_data_sets(config.datasets['mnist']['origin_dir_path'], one_hot=True)
    images = mnist.test.images
    labels = mnist.test.labels
    images_classify_data = [[], [], [], [], [], [], [], [], [], []]
    # 按照0-9将测试数据分开
    for i in range(len(labels)):
        print(i)
        pos = list(labels[i]).index(1)
        images_classify_data[pos].append(images[i])

    for i in range(len(images_classify_data)):
        save_path = os.path.join(config.datasets['mnist']['test_dir_path'], str(i) + '.npy')
        np.save(save_path, images_classify_data[i])


def visual_cifar10(data):
    data = data.reshape((3, 32, 32))
    data = np.concatenate([data[0].reshape(32, 32, 1), data[1].reshape(32, 32, 1), data[2].reshape(32, 32, 1)], -1)
    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.show()


def visual_mnist(data):
    data = data.reshape(28, 28)
    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plotwindow.imshow(data, cmap='gray')
    plt.show()


if __name__ == '__main__':
    print(config.exp['shape'])
    # 建立好对应的目录
    mkdir()
    # 整理cifar10的数据 分为测试集和训练集合
    save_cifar10_train()
    save_cifar10_test()
