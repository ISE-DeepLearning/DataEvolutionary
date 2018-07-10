# -*- coding: utf-8 -*-

import math
import os
import random

import numpy as np

import image_process.image_util as util

import matplotlib.pyplot as plt

import mnist.data_process.config as config

"""
input: matrix(n*784)
output: matrix(2C(2,n),784)
note: 1. It will be better if n is relative small(n>=2)
      2. The original pics should have been moved in center.
"""


def generate_horizontal(original):
    m, n = 0, len(original)
    result = np.zeros((n * (n - 1), 784))
    for i in range(n):
        for offset in range(1, n - i):
            pic1, pic2 = original[i].reshape(28, 28), original[i + offset].reshape(28, 28)
            temp1 = util.cut2part(pic1, 'horizontal')
            temp2 = util.cut2part(pic2, 'horizontal')
            result[m] = util.join2part(temp1[0], temp2[1], 'vertical').flatten()
            result[m + 1] = util.join2part(temp2[0], temp1[1], 'vertical').flatten()
            m += 2
    return result


"""
input: matrix(n*784)
output: matrix(2C(2,n),784)
note: 1. It will be better if n is relative small(n>=2)
      2. The original pics should have been moved in center.
"""


def generate_vertical(original):
    m, n = 0, len(original)
    result = np.zeros((n * (n - 1), 784))
    for i in range(n):
        for offset in range(1, n - i):
            pic1, pic2 = original[i].reshape(28, 28), original[i + offset].reshape(28, 28)
            temp1 = util.cut2part(pic1, 'vertical')
            temp2 = util.cut2part(pic2, 'vertical')
            result[m] = util.join2part(temp1[0], temp2[1], 'horizontal').flatten()
            result[m + 1] = util.join2part(temp2[0], temp1[1], 'horizontal').flatten()
            m += 2
    return result


"""
input: matrix(n*784)
output: matrix(2C(2,n),784)
note: 1. It will be better if n is relative small(n>=2)
      2. The original pics should have been moved in center.
"""


def generate_mix(original, mode='max'):
    m, n = 0, len(original)
    result = np.zeros((n * (n - 1) / 2, 784))
    for i in range(n):
        for offset in range(1, n - i):
            pic1, pic2 = original[i].reshape(28, 28), original[i + offset].reshape(28, 28)
            center1, angle1 = util.find_center(pic1), util.cal_angle(pic1)
            center2, angle2 = util.find_center(pic2), util.cal_angle(pic2)
            change_angle = math.fabs(angle1 - angle2)
            if angle1 > angle2:
                pic2 = util.rotate(pic2, change_angle, fill=0)
            else:
                pic1 = util.rotate(pic1, change_angle, fill=0)
            pic1 = util.move(pic1, x=center2[0] - center1[0], y=center2[1] - center1[1], shape=(28, 28))
            result[m] = util.mix(pic1, pic2, mode=mode).flatten()
            m += 1
    return result


def generate_all(original):
    result = generate_horizontal(original)
    result = np.vstack((result, generate_vertical(original)))
    result = np.vstack((result, generate_mix(original, 'max')))
    # result = np.vstack((result, generate_mix(original, 'min')))
    result = np.vstack((result, generate_mix(original, 'average')))
    # result = np.vstack((result, generate_mix(original, 'add')))
    return result


def load_data(number, n):
    # address = '../mnist/training_npy/' + str(number) + '.npy'
    address = '../training_npy/' + str(number) + '.npy'
    return random.sample(np.load(address), n)


def save_data(image, number, mode):
    path = os.path.join('../experiment/', config.exp_index, mode)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, str(number) + '.npy')
    np.save(path, image)


def vision(image_data):
    im = image_data
    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plotwindow.imshow(im, cmap='gray')
    plt.show()


if __name__ == '__main__':
    for i in range(10):
        data = load_data(i, 10)
        all = generate_all(data)
        save_data(data, i, 'original')
        save_data(all, i, 'evolution')
