# -*- coding: utf-8 -*-
import numpy as np
import foolbox
import keras
from keras.models import load_model
import json
from PIL import Image
import config
from foolbox.criteria import TargetClassProbability, TargetClass
import math
import matplotlib
import os
import experiment as exp
# matplotlib.use('Agg')
# from matplotlib import pyplot as plt


import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# 进行配置，使用30%的GPU
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=gpu_config)

# 设置session
KTF.set_session(session)


# 计算成绩
def cal_score(original_data, compose_data):
    test_count = 0
    shape = config.exp['shape']
    pixels = shape[0] * shape[1] * shape[2]
    for i in range(0, pixels):
        x = (original_data.reshape(pixels)[i] * 255).astype(np.uint8)
        y = (compose_data.reshape(pixels)[i] * 255).astype(np.uint8)
        temp = (int(x) - int(y)) ** 2
        test_count = test_count + temp
    # print(test_count)
    mse_pow = float(test_count) / float(pixels)
    mse = math.sqrt(mse_pow)
    # print(mse)
    # div = mse / 70
    score = 100 / (1 + math.pow(math.e, (mse - 70) / 15))
    # count = np.sum((original_data.reshape(28, 28) - compose_data.reshape(28, 28)) ** 2)
    # # 平方和 / 784
    # mse_pow = float(count) / float(len(original_data.flatten()))
    # mse = math.sqrt(mse_pow)
    # score = 100 / (1 + math.pow(math.e, (mse - 50) / 15))
    return score


# 攻击函数
def start_attack(foolmodel, image, label, threshold=90):
    advs = []

    for i in range(config.exp['label_num']):
        if i is not label:
            # 希望模型能将图片识别成i ~定向
            criterion = TargetClass(i)
            # 提供三种主流的对抗攻击方式 {基于决策的，基于梯度的，基于分数的}
            # Decision-based attacks
            attack1_1 = foolbox.attacks.AdditiveUniformNoiseAttack(foolmodel, criterion=criterion)
            attack1_2 = foolbox.attacks.AdditiveGaussianNoiseAttack(foolmodel, criterion=criterion)
            # Gradient-based attacks
            attack2 = foolbox.attacks.FGSM(foolmodel, criterion=criterion)
            attack3 = foolbox.attacks.SaliencyMapAttack(foolmodel, criterion=criterion)
            # Score-based attacks 不采用,数据都很差 评分都很低
            # attack4 = foolbox.attacks.LocalSearchAttack(foolmodel, criterion=criterion)
            # 使用第一种第二种攻击方案
            for eps in range(100, 1000, 100):
                # 1. 使用Decision-based attacks
                print(eps)
                adv1_1 = attack1_1(image, label, epsilons=eps)
                adv1_2 = attack1_2(image, label, epsilons=eps)
                if adv1_1 is not None and cal_score(image, adv1_1) >= threshold:
                    # 将高于threshold的数据保留下来，高的不需要保留
                    advs.append(adv1_1)
                if adv1_2 is not None and cal_score(image, adv1_2) >= threshold:
                    # 将高于threshold的数据保留下来，高的不需要保留
                    advs.append(adv1_2)

            # 使用第三种攻击方案
            eps_set = [10, 100, 200, 300, 500, 800]
            temp_score = 0
            for eps in eps_set:
                print(eps)
                # 2. 使用Gradient-based attacks
                adv2 = attack2(image, label, epsilons=eps)
                if adv2 is not None:
                    score = cal_score(image, adv2)
                    if score >= threshold and temp_score != score:
                        advs.append(adv2)
                        temp_score = score

            # 使用第四种攻击方案
            temp_score = 0
            for theta in np.arange(0.1, 1.1, 0.1):
                print(theta)
                # 2. 使用Gradient-based attacks
                adv3 = attack3(image, label, theta=theta)
                if adv3 is not None:
                    score = cal_score(image, adv3)
                    if score >= threshold and temp_score != score:
                        advs.append(adv3)
                        temp_score = score

    # 返回生成的攻击样本，原始标签以及对抗之后的标签，用于保存
    # labels = foolmodel.batch_predictions(np.array(advs))
    # labels = np.argmax(labels, axis=1)
    # original_labels = np.zeros(shape=(len(labels),))
    # original_labels[:] = label

    return np.array(advs)


if __name__ == '__main__':
    # 获取original的均值
    train_orignal_datas = []
    shape = config.exp['shape']
    for i in range(0, config.exp['label_num']):
        model_dir_path = './model/' + config.exp['dataset'] + '/cnn/' + str(config.exp['sample_every_label'])
        datas = np.load(os.path.join(model_dir_path, str(i) + '.npy'))
        print(datas.shape)
        train_orignal_datas.append(datas)
    images = np.reshape(np.array(train_orignal_datas), (-1, shape[0], shape[1], shape[2]))
    images = images.transpose((0, 2, 3, 1))
    images = exp.pre_mean(np.array(images), np.shape(images))
    # 图片均值
    mean_data = exp.global_mean_data
    # cifar的bounds
    bounds = (0 - np.max(mean_data), 1 - np.min(mean_data))
    print(bounds)
    for i in range(0, config.exp['label_num']):
        adv_datas = []
        adv_labels = []
        # 加载i label下面的图片数据
        model_dir_path = './model/' + config.exp['dataset'] + '/cnn/' + str(config.exp['sample_every_label'])
        datas = np.load(os.path.join(model_dir_path, str(i) + '.npy'))
        # 加载对应的粗糙模型
        model = load_model(os.path.join(model_dir_path, 'model.hdf5'))
        # 开始对每个样本找对应的模型数据
        keras.backend.set_learning_phase(0)
        shape = config.exp['shape']
        for j in range(len(datas)):
            print(str(i) + '-' + str(j) + '/' + str(len(datas)))
            foolmodel = foolbox.models.KerasModel(model, bounds=bounds, preprocessing=(0, 1))
            # print(foolmodel.bounds())
            # shape = config.exp['shape']
            # print(datas[i].shape)
            image = np.reshape(datas[j], newshape=shape)
            # print(image.shape)
            image = np.transpose(image, (1, 2, 0))
            # 均值化
            for channel in range(image.shape[-1]):
                image[:, :, channel] -= mean_data[channel]
            # 原始 label 为 i 生成对应的label
            # print(i)
            # 开始攻击 批量生成大量的关于本图片的images
            # 最后一个threshold表示什么样子的对抗样本我们才保存
            advs = start_attack(foolmodel, image, i, threshold=90)
            print(len(advs))
            adv_datas.extend(advs)
            # adv_labels.extend(labels)
            # plt.imshow(adv.reshape(28, 28))
            # plt.imsave('adversial.jpg', adv.reshape(28, 28), format="jpg", cmap='gray')
        # saving data
        print('saving data...')
        if len(adv_datas) > 0:
            adv_datas = np.reshape(adv_datas, (-1, shape[0] * shape[1] * shape[2]))
            np.save(os.path.join(model_dir_path, 'adv_' + str(i) + '.npy'), adv_datas)
            labels = model.predict(np.reshape(adv_datas, (-1, shape[1], shape[2], shape[0])))
            labels = np.argmax(labels, axis=1)
            np.save(os.path.join(model_dir_path, 'adv_' + str(i) + '_label.npy'), adv_labels)
    print('The end ...')
