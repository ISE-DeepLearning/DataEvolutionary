# -*- coding:utf-8 -*-
import os
import numpy as np
import mnist.data_process.config as config


def original_test_data():
    images = []
    labels = []
    for i in range(10):
        npy_path = os.path.join('../test/npy', str(i) + '.npy')
        image_set = np.load(npy_path)
        label_set = np.zeros(shape=(len(image_set), 10))
        label_set[:, i] = 1
        images.extend(image_set)
        labels.extend(label_set)
    # images = np.row_stack((images, images))
    # labels = np.row_stack((labels, labels))
    return images, labels


def evolution_data():
    images = []
    labels = []
    for i in range(10):
        original_path = os.path.join('../experiment/', config.exp_index, 'original', str(i) + '.npy')
        original_set = np.load(original_path)
        original_label_set = np.zeros(shape=(len(original_set), 10))
        original_label_set[:, i] = 1

        evolution_path = os.path.join('../experiment/', config.exp_index, 'evolution', str(i) + '.npy')
        evolution_set = np.load(evolution_path)
        evolution_label_set = np.zeros(shape=(len(evolution_set), 10))
        evolution_label_set[:, i] = 1

        images.extend(original_set)
        images.extend(evolution_set)
        labels.extend(original_label_set)
        labels.extend(evolution_label_set)
    return images, labels


def repeat_data():
    images = []
    labels = []
    for i in range(10):
        original_path = os.path.join('../experiment/', config.exp_index, 'original', str(i) + '.npy')
        original_set = np.load(original_path)
        original_label_set = np.zeros(shape=(len(original_set), 10))
        original_label_set[:, i] = 1

        evolution_path = os.path.join('../experiment/', config.exp_index, 'evolution', str(i) + '.npy')
        evolution_set = np.load(evolution_path)

        m = len(evolution_set)
        n = len(original_set)
        e = np.eye(n)
        if n > m:
            e = e[:m]
        else:
            k = m / n
            r = m % n
            for j in range(k - 1):
                e = np.vstack((e, np.eye(n)))
            np.vstack((e, np.eye(n)[:r]))

        evolution_set = np.dot(e, original_set)
        evolution_label_set = np.zeros(shape=(len(evolution_set), 10))
        evolution_label_set[:, i] = 1

        images.extend(original_set)
        images.extend(evolution_set)
        labels.extend(original_label_set)
        labels.extend(evolution_label_set)
    return images, labels


if __name__ == '__main__':
    print(np.shape(evolution_data()[0]), np.shape(evolution_data()[1]))
    print(np.shape(repeat_data()[0]), np.shape(repeat_data()[1]))
