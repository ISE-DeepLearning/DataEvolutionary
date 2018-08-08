# -*- coding=utf-8 -*-
import numpy as np
import random
import cross
import variation
import config
import os

'''
获取训练数据
'''

'''
获得原始数据，num是每种样本的张数
'''

shape = config.exp['shape']
dataset = config.exp['dataset']
upper = config.exp['upper']


def get_origin(i, num):
    result = []
    address = os.path.join('./', dataset, 'training_npy', str(i) + '.npy')
    data = random.sample(list(np.load(address)), num)
    result += data
    return np.array(result) / upper


# n*(n-1)
def get_horizontal(origin):
    result = []
    n = len(origin)
    for i in range(n):
        for offset in range(1, n - i):
            pic1, pic2 = origin[i].reshape(shape), origin[i + offset].reshape(shape)
            new1, new2 = cross.exchange(pic1, pic2, (0, 0), shape[1], shape[1] // 2)
            result.append(new1.flatten())
            result.append(new2.flatten())
    return np.array(result)


# n*(n-1)
def get_vertical(origin):
    result = []
    n = len(origin)
    for i in range(n):
        for offset in range(1, n - i):
            pic1, pic2 = origin[i].reshape(shape), origin[i + offset].reshape(shape)
            new1, new2 = cross.exchange(pic1, pic2, (0, 0), shape[1] // 2, shape[1])
            result.append(new1.flatten())
            result.append(new2.flatten())
    return np.array(result)


# n*2*times
def get_exchange(origin, times=1):
    n = len(origin)
    result = []
    for i in range(times * n):
        [pic1, pic2] = random.sample(list(origin), 2)
        [pic1, pic2] = [pic1.reshape(shape), pic2.reshape(shape)]
        random_point = (int(random.random() * shape[1]), int(random.random() * shape[1]))
        random_w, random_h = int(random.random() * shape[1]), int(random.random() * shape[1])
        new1, new2 = cross.exchange(pic1, pic2, random_point, random_w, random_h)
        result.append(new1.flatten())
        result.append(new2.flatten())
    return np.array(result)


# n*(n-1)/2
def get_mix(origin, mode='average'):
    result = []
    n = len(origin)
    for i in range(n):
        for offset in range(1, n - i):
            pic1, pic2 = origin[i].reshape(shape), origin[i + offset].reshape(shape)
            new = cross.mix_up(pic1, pic2, mode)
            result.append(new.flatten())
    return np.array(result)


# n*times
def get_rotated(origin, times=2):
    n = len(origin)
    result = []
    for i in range(times * n):
        [pic] = random.sample(list(origin), 1)
        pic = pic.reshape(shape)
        random_angle = random.random() * 45
        new = variation.rotate(pic, random_angle)
        result.append(new.flatten())
    return np.array(result)


def get_data(origin):
    modes = config.exp['mode']
    result = []
    for mode in modes:
        if mode == 'all':
            result += list(get_horizontal(origin))
            result += list(get_vertical(origin))
            result += list(get_exchange(origin))
            result += list(get_rotated(origin))
            result += list(get_mix(origin, 'average'))
            result += list(get_mix(origin, 'max'))
            result += list(get_mix(origin, 'min'))
            result += list(get_mix(origin, 'add'))
        elif mode == 'horizontal':
            result += list(get_horizontal(origin))
        elif mode == 'vertical':
            result += list(get_vertical(origin))
        elif mode == 'exchange':
            result += list(get_exchange(origin))
        elif mode == 'rotate':
            result += list(get_rotated(origin))
        else:
            result += list(get_mix(origin, mode))
    return np.array(result)


def save(data, label, mode):
    address = os.path.join(config.exp['exp_dir_path'], config.exp['exp_index'], mode)
    if not os.path.exists(address):
        os.makedirs(address)
    path = os.path.join(address, str(label) + '.npy')
    np.save(path, data)


if __name__ == '__main__':
    for i in range(config.exp['label_num']):
        origin = get_origin(i, config.exp['sample_every_label'])
        evolution = get_data(origin)
        save(origin, i, 'origin')
        save(evolution, i, 'evolution')
