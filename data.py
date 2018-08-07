# -*- coding=utf-8 -*-
import numpy as np
import random
import cross
import variation
import config
import os
import save_data

'''
获取训练数据
'''

'''
获得原始数据，num是每种样本的张数
'''


def get_origin(i, num, dataset='mnist'):
    result = []
    address = os.path.join('./', dataset, 'training_npy', str(i) + '.npy')
    data = random.sample(list(np.load(address)), num)
    result += data
    if dataset == 'mnist':
        return np.array(result)
    else:
        return np.array(result) / 255


# n*(n-1)
def get_horizontal(origin, dataset='mnist'):
    result = []
    n = len(origin)
    for i in range(n):
        for offset in range(1, n - i):
            if dataset == 'mnist':
                pic1, pic2 = origin[i].reshape(1, 28, 28), origin[i + offset].reshape(1, 28, 28)
                new1, new2 = cross.exchange(pic1, pic2, (0, 0), 28, 14)
            else:
                pic1, pic2 = origin[i].reshape(3, 32, 32), origin[i + offset].reshape(3, 32, 32)
                new1, new2 = cross.exchange(pic1, pic2, (0, 0), 32, 16)
            result.append(new1.flatten())
            result.append(new2.flatten())
    return np.array(result)


# n*(n-1)
def get_vertical(origin, dataset='mnist'):
    result = []
    n = len(origin)
    for i in range(n):
        for offset in range(1, n - i):
            if dataset == 'mnist':
                pic1, pic2 = origin[i].reshape(1, 28, 28), origin[i + offset].reshape(1, 28, 28)
                new1, new2 = cross.exchange(pic1, pic2, (0, 0), 14, 28)
            else:
                pic1, pic2 = origin[i].reshape(3, 32, 32), origin[i + offset].reshape(3, 32, 32)
                new1, new2 = cross.exchange(pic1, pic2, (0, 0), 16, 32)
            result.append(new1.flatten())
            result.append(new2.flatten())
    return np.array(result)


# n*2*times
def get_exchange(origin, dataset='mnist', times=1):
    n = len(origin)
    result = []
    for i in range(times * n):
        [pic1, pic2] = random.sample(list(origin), 2)
        if dataset == 'mnist':
            [pic1, pic2] = [pic1.reshape(1, 28, 28), pic2.reshape(1, 28, 28)]
            random_point = (int(random.random() * 28), int(random.random() * 28))
            random_w, random_h = int(random.random() * 28), int(random.random() * 28)
        else:
            [pic1, pic2] = [pic1.reshape(3, 32, 32), pic2.reshape(3, 32, 32)]
            random_point = (int(random.random() * 28), int(random.random() * 28))
            random_w, random_h = int(random.random() * 28), int(random.random() * 28)
        new1, new2 = cross.exchange(pic1, pic2, random_point, random_w, random_h)
        result.append(new1.flatten())
        result.append(new2.flatten())
    return np.array(result)


# n*(n-1)/2
def get_mix(origin, mode='average', dataset='mnist'):
    result = []
    n = len(origin)
    for i in range(n):
        for offset in range(1, n - i):
            if dataset == 'mnist':
                pic1, pic2 = origin[i].reshape(1, 28, 28), origin[i + offset].reshape(1, 28, 28)
            else:
                pic1, pic2 = origin[i].reshape(3, 32, 32), origin[i + offset].reshape(3, 32, 32)
            new = cross.mix_up(pic1, pic2, mode)
            result.append(new.flatten())
    return np.array(result)


# n*times
def get_rotated(origin, dataset='mnist', times=2):
    n = len(origin)
    result = []
    for i in range(times * n):
        [pic] = random.sample(list(origin), 1)
        if dataset == 'mnist':
            pic = pic.reshape(1, 28, 28)
        else:
            pic = pic.reshape(3, 32, 32)
        random_angle = random.random() * 45
        new = variation.rotate(pic, random_angle)
        result.append(new.flatten())
    return np.array(result)


def get_data(origin, dataset='mnist'):
    modes = config.exp['mode']
    result = []
    for mode in modes:
        if mode == 'all':
            result += list(get_horizontal(origin, dataset))
            result += list(get_vertical(origin, dataset))
            result += list(get_exchange(origin, dataset))
            result += list(get_rotated(origin, dataset))
            result += list(get_mix(origin, 'average', dataset))
            result += list(get_mix(origin, 'max', dataset))
            result += list(get_mix(origin, 'min', dataset))
            result += list(get_mix(origin, 'add', dataset))
        elif mode == 'horizontal':
            result += list(get_horizontal(origin, dataset))
        elif mode == 'vertical':
            result += list(get_vertical(origin, dataset))
        elif mode == 'exchange':
            result += list(get_exchange(origin, dataset))
        elif mode == 'rotate':
            result += list(get_rotated(origin, dataset))
        else:
            result += list(get_mix(origin, mode, dataset))
    return np.array(result)


def save(data, label, mode):
    address = os.path.join(config.exp['exp_dir_path'], config.exp['exp_index'], mode)
    if not os.path.exists(address):
        os.makedirs(address)
    path = os.path.join(address, str(label) + '.npy')
    np.save(path, data)


if __name__ == '__main__':
    for i in range(10):
        origin = get_origin(i, 10, config.exp['dataset'])
        evolution = get_data(origin, config.exp['dataset'])
        save(origin, i, 'origin')
        save(evolution, i, 'evolution')
