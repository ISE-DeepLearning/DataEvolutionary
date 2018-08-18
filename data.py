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


# 从一个比较大的数据集中选10%
def get_expand_and_origin(i, num, times=10):
    result = []
    expand_result = []
    address = os.path.join('./', dataset, 'training_npy', str(i) + '.npy')
    expand_data = random.sample(list(np.load(address)), num * times)
    expand_result += expand_data
    data = random.sample(expand_data, num)
    result += data
    return np.array(expand_result) / upper, np.array(result) / upper


# TODO 获取model文件夹里面对应的数据
# TODO 数据已经除以了upper了不需要再除以upper
def get_mutation_expand_and_origin(i, num):
    result = []
    expand_result = []
    base_path = os.path.join('./model', dataset, 'cnn/', str(num))
    expand_data = np.load(os.path.join(base_path, 'expand_' + str(i) + '.npy'))
    data = np.load(os.path.join(base_path, str(i) + '.npy'))
    expand_result += expand_data
    result += data
    return np.array(expand_result), np.array(result)


# n*(n-1)
def get_horizontal(origin, num=None):
    result = []
    n = len(origin)
    count = 0
    for i in range(n):
        for offset in range(1, n - i):
            pic1, pic2 = origin[i].reshape(shape), origin[i + offset].reshape(shape)
            new1, new2 = cross.exchange(pic1, pic2, (0, 0), shape[1], shape[1] // 2)
            result.append(new1.flatten())
            result.append(new2.flatten())
            count += 0
            if num is not None and count >= num:
                return np.array(result)
    return np.array(result)


# n*(n-1)
def get_vertical(origin, num=None):
    result = []
    n = len(origin)
    count = 0
    for i in range(n):
        for offset in range(1, n - i):
            pic1, pic2 = origin[i].reshape(shape), origin[i + offset].reshape(shape)
            new1, new2 = cross.exchange(pic1, pic2, (0, 0), shape[1] // 2, shape[1])
            result.append(new1.flatten())
            result.append(new2.flatten())
            count += 2
            if num is not None and count >= num:
                return np.array(result)
    return np.array(result)


# n*2*times
# 随机内切两张图片的方块，进行互换 crossover-exchange
def get_exchange(origin, times=10, num=None):
    if not num:
        num = len(origin) * times
    result = []
    count = 0
    for i in range(num):
        [pic1, pic2] = random.sample(list(origin), 2)
        [pic1, pic2] = [pic1.reshape(shape), pic2.reshape(shape)]
        random_point = (int(random.random() * shape[1]), int(random.random() * shape[1]))
        random_w, random_h = int(random.random() * shape[1]), int(random.random() * shape[1])
        new1, new2 = cross.exchange(pic1, pic2, random_point, random_w, random_h)
        result.append(new1.flatten())
        result.append(new2.flatten())
        count += 2
        if count >= num:
            return np.array(result)
    return np.array(result)


# n*(n-1)/2
# 如果num为none就是全生成
# 如果num为数字，就生成对应数量的样本
def get_mix(origin, mode='average', num=None):
    result = []
    n = len(origin)
    count = 0
    for i in range(n):
        for offset in range(1, n - i):
            pic1, pic2 = origin[i].reshape(shape), origin[i + offset].reshape(shape)
            new = cross.mix_up(pic1, pic2, mode)
            result.append(new.flatten())
            count += 1
            # 如果已经生成足够数量的
            if num is not None and count >= num:
                return np.array(result)
    return np.array(result)


# n*times
def get_rotated(origin, times=10):
    n = len(origin)
    result = []
    for i in range(times * n):
        [pic] = random.sample(list(origin), 1)
        pic = pic.reshape(shape)
        random_angle = random.random() * 45
        new = variation.rotate(pic, random_angle)
        result.append(new.flatten())
    return np.array(result)


# TODO
def get_attacks(i, nums=None):
    # origin_data = []
    # expand_data = []
    # for i in range(config.exp['label_num']):
    # origial_path =
    attack_path = './model/' + config.exp['dataset'] + '/cnn/' + str(config.exp['sample_every_label']) + '/adv_' + str(
        i) + '.npy'
    attack_data = np.load(attack_path)
    if nums is not None and nums < len(attack_data):
        attack_data = random.sample(list(attack_data), nums)
    return attack_data


def get_data(origin, num=None):
    modes = config.exp['mode']
    result = []
    # 目前总共五种扩增手段
    all_type_nums = 5

    for mode in modes:
        if mode == 'all':
            result += list(get_horizontal(origin))
            result += list(get_vertical(origin))
            result += list(get_exchange(origin))
            result += list(get_mix(origin, 'average'))
            # result += list(get_attack(origin))
            # result += list(get_mix(origin, 'max'))
            # result += list(get_mix(origin, 'min'))
            # result += list(get_mix(origin, 'add'))
        elif mode == 'crossover':
            # crossover的手段集合
            result += list(get_horizontal(origin))
            result += list(get_vertical(origin))
            result += list(get_exchange(origin))
            result += list(get_mix(origin, 'average'))
        elif mode == 'mutation':
            result += list(get_rotated(origin, times=10))
            result += list(get_attacks(0, times=10))
        elif mode == 'attack':
            result += list(get_attacks(0, times=10))
        elif mode == 'rotate':
            result += list(get_rotated(origin, times=10))
        elif mode == 'horizontal':
            result += list(get_horizontal(origin))
        elif mode == 'vertical':
            result += list(get_vertical(origin))
        elif mode == 'exchange':
            result += list(get_exchange(origin))
        else:
            result += list(get_mix(origin, mode))
    return np.array(result)


def save(data, label, mode):
    address = os.path.join(config.exp['exp_dir_path'], config.exp['exp_index'], mode)
    if not os.path.exists(address):
        os.makedirs(address)
    path = os.path.join(address, str(label) + '.npy')
    np.save(path, data)


def process_data(exp):
    # 由于attack样本太难生成暂时先同其余的扩增方案分开来写
    for i in range(exp['label_num']):
        if exp['mode'] == 'attack' or exp['mode'] == 'mutation':
            # TODO
            expand, origin = get_mutation_expand_and_origin(i, exp['sample_every_label'])
            evolution = get_attacks(i, len(origin) * 10)
            save(expand, i, 'expand')
            save(origin, i, 'origin')
            save(evolution, i, 'evolution')
        else:
            expand, origin = get_expand_and_origin(i, exp['sample_every_label'])
            # config.exp('label')
            num = None
            # 默认是生成全部的扩增数据的，但是支持生成固定数量的扩增数据
            if 'evolution_every_label' in exp.keys():
                num = exp['evolution_every_label']
            evolution = get_data(origin, num)
            save(expand, i, 'expand')
            save(origin, i, 'origin')
            save(evolution, i, 'evolution')


if __name__ == '__main__':
    for i in range(config.exp['label_num']):
        expand, origin = get_expand_and_origin(i, config.exp['sample_every_label'])
        # config.exp('label')
        num = None
        # 默认是生成全部的扩增数据的，但是支持生成固定数量的扩增数据
        if 'evolution_every_label' in config.exp.keys():
            num = config.exp['evolution_every_label']
        evolution = get_data(origin, num)
        save(expand, i, 'expand')
        save(origin, i, 'origin')
        save(evolution, i, 'evolution')
