import os
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam

from PIL import Image
import config
import json
import data

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# 进行配置，使用30%的GPU
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=gpu_config)

# 设置session
KTF.set_session(session)

global_mean_data = [0, 0, 0]


def get_origin(base_path, max_num=None):
    images = []
    labels = []
    shape = config.exp['shape']
    for i in range(config.exp['label_num']):
        image_set = np.load(os.path.join(base_path, str(i) + '.npy'))
        label_set = np.zeros(shape=(len(image_set), config.exp['label_num']))
        label_set[:, i] = 1
        images.extend(image_set)
        labels.extend(label_set)
    images = np.reshape(np.array(images), (-1, shape[0], shape[1], shape[2]))
    images = images.transpose((0, 2, 3, 1))
    # 预处理 彩色图片进行均值化操作
    images = pre_mean(images, np.shape(images))
    return images, np.array(labels)


def get_original_data(max_num=None):
    images = []
    labels = []
    expand_images = []
    expand_labels = []
    shape = config.exp['shape']
    for i in range(config.exp['label_num']):
        expand_image_set, image_set = data.get_expand_and_origin(i, config.exp['sample_every_label'])
        label_set = np.zeros(shape=(len(image_set), config.exp['label_num']))
        expand_label_set = np.zeros(shape=(len(expand_image_set), config.exp['label_num']))
        label_set[:, i] = 1
        expand_label_set[:, i] = 1
        images.extend(image_set)
        expand_images.extend(expand_image_set)
        labels.extend(label_set)
        expand_labels.extend(expand_label_set)
        model_dir_path = './model/' + config.exp['dataset'] + '/cnn/' + str(config.exp['sample_every_label'])
        if not os.path.exists(model_dir_path):
            os.mkdir(model_dir_path)
        np.save(os.path.join(model_dir_path, str(i) + '.npy'), image_set)
        np.save(os.path.join(model_dir_path, 'expand_' + str(i) + '.npy'), expand_image_set)
    temp_images = images.copy()
    temp_labels = labels.copy()
    while max_num is not None and len(images) < max_num:
        images = images + temp_images
        labels = labels + temp_labels
    images = np.reshape(np.array(images), (-1, shape[0], shape[1], shape[2]))
    images = images.transpose((0, 2, 3, 1))
    # 预处理 彩色图片进行均值化操作
    images = pre_mean(images, np.shape(images))
    return images, np.array(labels)


def get_train_data():
    dataset = config.exp['dataset']
    shape = config.exp['shape']
    images = []
    labels = []
    path = config.datasets[dataset]['training_dir_path']
    for i in range(config.exp['label_num']):
        npy_path = os.path.join(path, str(i) + '.npy')
        image_set = np.load(npy_path)
        label_set = np.zeros(shape=(len(image_set), config.exp['label_num']))
        label_set[:, i] = 1
        images.extend(image_set)
        labels.extend(label_set)
    images = np.reshape(np.array(images), (-1, shape[0], shape[1], shape[2]))
    images = images.transpose((0, 2, 3, 1))
    images = images / config.exp['upper']
    # 预处理 彩色图片进行均值化操作
    images = pre_mean(images, np.shape(images))
    return images, np.array(labels)


def get_test_data():
    dataset = config.exp['dataset']
    shape = config.exp['shape']
    images = []
    labels = []
    path = config.datasets[dataset]['test_dir_path']
    for i in range(config.exp['label_num']):
        npy_path = os.path.join(path, str(i) + '.npy')
        image_set = np.load(npy_path)
        label_set = np.zeros(shape=(len(image_set), config.exp['label_num']))
        label_set[:, i] = 1
        images.extend(image_set)
        labels.extend(label_set)
    images = np.reshape(np.array(images), (-1, shape[0], shape[1], shape[2]))
    images = images.transpose((0, 2, 3, 1))
    # 预处理 彩色图片进行均值化操作
    images = images / config.exp['upper']

    # images = pre_mean(images, np.shape(images))
    return images, np.array(labels)


def origin_data():
    shape = config.exp['shape']
    images = []
    labels = []
    for i in range(config.exp['label_num']):
        original_path = os.path.join(config.exp['exp_dir_path'], config.exp['exp_index'], 'origin', str(i) + '.npy')
        original_set = np.load(original_path)
        original_label_set = np.zeros(shape=(len(original_set), config.exp['label_num']))
        original_label_set[:, i] = 1
        images.extend(original_set)
        labels.extend(original_label_set)
    images = np.reshape(np.array(images), (-1, shape[0], shape[1], shape[2]))
    images = images.transpose((0, 2, 3, 1))
    # 预处理 彩色图片进行均值化操作
    images = pre_mean(images, np.shape(images))
    return images, np.array(labels)


def expand_data():
    shape = config.exp['shape']
    images = []
    labels = []
    for i in range(config.exp['label_num']):
        expand_path = os.path.join(config.exp['exp_dir_path'], config.exp['exp_index'], 'expand', str(i) + '.npy')
        expand_set = np.load(expand_path)
        expand_label_set = np.zeros(shape=(len(expand_set), config.exp['label_num']))
        expand_label_set[:, i] = 1
        images.extend(expand_set)
        labels.extend(expand_label_set)
    images = np.reshape(np.array(images), (-1, shape[0], shape[1], shape[2]))
    images = images.transpose((0, 2, 3, 1))
    # 预处理 彩色图片进行均值化操作
    images = pre_mean(images, np.shape(images))
    return images, np.array(labels)


def repeat_data():
    shape = config.exp['shape']
    images = []
    labels = []
    for i in range(config.exp['label_num']):
        original_path = os.path.join(config.exp['exp_dir_path'], config.exp['exp_index'], 'origin', str(i) + '.npy')
        original_set = np.load(original_path)
        original_label_set = np.zeros(shape=(len(original_set), config.exp['label_num']))
        original_label_set[:, i] = 1

        evolution_path = os.path.join(config.exp['exp_dir_path'], config.exp['exp_index'], 'evolution', str(i) + '.npy')
        evolution_set = np.load(evolution_path)
        m = len(evolution_set)
        n = len(original_set)
        e = np.eye(n)
        if n > m:
            e = e[:m]
        else:
            k = m // n
            r = m % n
            for j in range(k - 1):
                e = np.vstack((e, np.eye(n)))
            np.vstack((e, np.eye(n)[:r]))
        evolution_set = np.dot(e, original_set)
        evolution_label_set = np.zeros(shape=(len(evolution_set), config.exp['label_num']))
        evolution_label_set[:, i] = 1

        images.extend(original_set)
        images.extend(evolution_set)
        labels.extend(original_label_set)
        labels.extend(evolution_label_set)

    images = np.reshape(np.array(images), (-1, shape[0], shape[1], shape[2]))
    images = images.transpose((0, 2, 3, 1))
    # 预处理 彩色图片进行均值化操作
    images = pre_mean(images, np.shape(images))
    return images, np.array(labels)


def evolution_data():
    shape = config.exp['shape']
    images = []
    labels = []
    for i in range(config.exp['label_num']):
        original_path = os.path.join(config.exp['exp_dir_path'], config.exp['exp_index'], 'origin', str(i) + '.npy')
        original_set = np.load(original_path)
        original_label_set = np.zeros(shape=(len(original_set), config.exp['label_num']))
        original_label_set[:, i] = 1

        evolution_path = os.path.join(config.exp['exp_dir_path'], config.exp['exp_index'], 'evolution', str(i) + '.npy')
        evolution_set = np.load(evolution_path)
        evolution_label_set = np.zeros(shape=(len(evolution_set), config.exp['label_num']))
        evolution_label_set[:, i] = 1

        images.extend(original_set)
        images.extend(evolution_set)
        labels.extend(original_label_set)
        labels.extend(evolution_label_set)
    images = np.reshape(np.array(images), (-1, shape[0], shape[1], shape[2]))
    images = images.transpose((0, 2, 3, 1))
    # 预处理 彩色图片进行均值化操作
    images = pre_mean(images, np.shape(images))
    return images, np.array(labels)


def train(model_save_path, train_images, train_labels, test_images, test_labels, nb_classes=10,
          batch_size=64, epochs=5):
    shape = config.exp['shape']
    input_tensor = Input((shape[1], shape[2], shape[0]))
    # 28*28
    temp = Conv2D(filters=32, kernel_size=(3, 3), padding='valid', use_bias=False)(input_tensor)
    temp = Activation('relu')(temp)
    # 26*26
    temp = MaxPooling2D(pool_size=(2, 2))(temp)
    # 13*13
    temp = Conv2D(filters=64, kernel_size=(3, 3), padding='valid', use_bias=False)(temp)
    temp = Activation('relu')(temp)
    # 11*11
    temp = MaxPooling2D(pool_size=(2, 2))(temp)
    # 5*5
    temp = Conv2D(filters=128, kernel_size=(3, 3), padding='valid', use_bias=False)(temp)
    temp = Activation('relu')(temp)
    # 3*3
    temp = Flatten()(temp)
    temp = Dropout(0.3)(temp)

    temp = Dense(nb_classes)(temp)
    output = Activation('softmax')(temp)

    model = Model(input=input_tensor, outputs=output)

    model.summary()

    sgd = SGD(lr=0.01, decay=1e-3, momentum=0.9, nesterov=True)
    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
                     validation_data=(test_images, test_labels))
    # Y_pred = model.predict(X_test, verbose=0)
    score = model.evaluate(test_images, test_labels, verbose=0)
    print('测试集 score(val_loss): %.4f' % score[0])
    print('测试集 accuracy: %.4f' % score[1])
    # result.append({'path': model_save_path, 'val_loss': score[0], 'acc': score[1]})
    model.save(model_save_path)
    return hist


def train_rough_mnist_model(epochs=5):
    model_dir_path = './model/mnist/cnn/' + str(config.exp['sample_every_label'])
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    test_data, test_label = get_test_data()
    # 数量修改config 由小变大 50 100 200 300 500 1000
    train_data, train_label = get_original_data()
    print(train_data.shape)
    hist = train(os.path.join(model_dir_path, 'model.hdf5'), train_data, train_label, test_data, test_label,
                 epochs=epochs)
    # save data
    with open(os.path.join(model_dir_path, 'cnn_mnist_rough_model_result.json'), 'w') as outfile:
        json.dump(hist.history, outfile,
                  ensure_ascii=False)
        outfile.write('\n')


def train_rough_cifar10_model(batch_size=128, epochs=5):
    print(config.exp)
    model_dir_path = './model/cifar10/cnn/' + str(config.exp['sample_every_label'])
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    test_data, test_label = get_test_data()
    # 数量修改config 由小变大 50 100 200 300 500 1000
    train_data, train_label = get_original_data()
    for i in range(train_data.shape[-1]):
        test_data[:, :, :, i] -= global_mean_data[i]
    # print(train_data.shape)
    hist = train(os.path.join(model_dir_path, 'model.hdf5'), train_data, train_label, test_data, test_label,
                 batch_size=batch_size, epochs=epochs)
    # save data
    with open(os.path.join(model_dir_path, 'cnn_cifar10_rough_model_result.json'), 'w') as outfile:
        json.dump(hist.history, outfile,
                  ensure_ascii=False)
        outfile.write('\n')


# 均值化预处理针对cifar数据集
def pre_mean(datas, shape):
    # 如果有三个channel
    if shape[-1] == 3:
        # 进行均值化
        mean_data = np.zeros(shape=(shape[-1],))
        nums = shape[0] * shape[1] * shape[2]
        for i in range(shape[-1]):
            mean_data[i] = np.sum(datas[:, :, :, i]) / nums
            global_mean_data[i] = mean_data[i]
            datas[:, :, i] -= mean_data[i]
            print(mean_data[i] * 255)
    return datas


def do_exp(batch_size=64, epochs=10):
    for time in range(config.exp['repeat_times']):
        print(config.exp)
        # result = []
        mean_data_for_save = {}
        test_images, test_label = get_test_data()
        model_path = os.path.join(config.exp['exp_dir_path'], config.exp['exp_index'], 'model', str(time))
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # origin
        train_data, train_label = origin_data()
        test_data = test_images.copy()
        mean_data_for_save['origin_mean'] = global_mean_data.copy()
        for i in range(test_data.shape[-1]):
            test_data[:, :, :, i] -= global_mean_data[i]
        # origin训练历史
        hist_origin = train(
            os.path.join(model_path, 'repeat_samples_model.hdf5'),
            train_data,
            train_label,
            test_images=test_data, test_labels=test_label, batch_size=batch_size, epochs=epochs)

        # expand
        train_data, train_label = expand_data()
        test_data = test_images.copy()
        mean_data_for_save['expand_mean'] = global_mean_data.copy()
        for i in range(test_data.shape[-1]):
            test_data[:, :, :, i] -= global_mean_data[i]
        # expand训练历史
        hist_expand = train(
            os.path.join(model_path, 'repeat_samples_model.hdf5'),
            train_data,
            train_label,
            test_images=test_data, test_labels=test_label, batch_size=batch_size, epochs=epochs)

        # repeat
        train_data, train_label = repeat_data()
        test_data = test_images.copy()
        mean_data_for_save['repeat_mean'] = global_mean_data.copy()
        for i in range(test_data.shape[-1]):
            test_data[:, :, :, i] -= global_mean_data[i]
        # repeat 训练历史
        hist_repeat = train(
            os.path.join(model_path, 'repeat_samples_model.hdf5'),
            train_data,
            train_label,
            test_images=test_data, test_labels=test_label, batch_size=batch_size, epochs=epochs)

        # evolution
        train_data, train_label = evolution_data()
        test_data = test_images.copy()
        mean_data_for_save['evolution_mean'] = global_mean_data.copy()
        for i in range(test_data.shape[-1]):
            test_data[:, :, :, i] -= global_mean_data[i]
        # evolution 训练历史
        hist_evolution = train(
            os.path.join(model_path, 'evolution_samples_model.hdf5'),
            train_data,
            train_label,
            test_images=test_data, test_labels=test_label, batch_size=batch_size, epochs=epochs)
        # save data
        with open(
                os.path.join(config.exp['exp_dir_path'], config.exp['exp_index'], 'cnn_result_' + str(time) + '.json'),
                'w') as outfile:
            json.dump({"hist_origin": hist_origin.history, "hist_expand": hist_expand.history,
                       "hist_repeat": hist_repeat.history, "hist_evolution": hist_evolution.history,
                       "mean_data": mean_data_for_save}, outfile,
                      ensure_ascii=False)
            outfile.write('\n')

        # print(train_data[900].shape)
        # image_data = train_data[900]
        # # image_data = train_data[1000].reshape((3, 32, 32))
        # # image_data = np.transpose(image_data, (1, 2, 0))
        # image = Image.fromarray((255 * image_data).astype(np.uint8))
        # image.save(str(time) + '.png')
        # print(train_label[900])


if __name__ == '__main__':
    # 训练粗糙模型 为了生成对抗样本
    # train_rough_mnist_model()

    # 从训练集合中将original的数据扩增10倍，可以理解为original的数据是从这个扩增了10倍之后的数据集合中抽取了10%作为样本

    # 进行一次实验
    # samples_per_label = [5, 10, 20, 30, 50, 100]
    # for i in range(6):
    #     config.exp['exp_index'] = 'exp' + str(i + 1)
    #     config.exp['sample_every_label'] = samples_per_label[i]
    #     data.process_data(config.exp)
    #     do_exp(batch_size=128, epochs=100)

    # 进行一次实验 进行attack操作的试验
    # samples_per_label = [5, 10, 20, 30, 50, 100]
    # for i in range(6):
    #     config.exp['exp_index'] = 'exp' + str(i + 1)
    #     config.exp['sample_every_label'] = samples_per_label[i]
    #     data.process_data(config.exp)
    #     do_exp()

    # 训练粗糙的cifar10数据集
    # samples_per_label = [5, 10, 20, 30, 50, 100]
    # for i in range(6):
    #     config.exp['sample_every_label'] = samples_per_label[i]
    #     train_rough_cifar10_model(epochs=100)

    # 训练粗糙的mnist数据集
    samples_per_label = [5, 10, 20, 30, 50, 100]
    for i in range(6):
        config.exp['sample_every_label'] = samples_per_label[i]
        train_rough_mnist_model(epochs=100)


    # 获取original的训练结果
    # do_orginal_exp()

    # 获取从train_data扩增为原来10倍的数据集来看数据表示
    # 训练某次实验不进行扩增 不进行repeat的数据 直接训练 50 100 200 300 500 1000数据
    # do_10times_origin_exp()

    # cifar_mean = np.array([125.31, 122.95, 113.87]) / 255
    # train_datas, train_labels = get_train_data()
    # test_datas, test_labels = get_test_data()
    # R_channel = 0
    # G_channel = 0
    # B_channel = 0
    # R_channel = R_channel + np.sum(train_datas[:, :, :, 0])
    # G_channel = G_channel + np.sum(train_datas[:, :, :, 1])
    # B_channel = B_channel + np.sum(train_datas[:, :, :, 2])
    # print(R_channel / (len(train_datas) * 32 * 32) * 255)
    # print(G_channel / (len(train_datas) * 32 * 32) * 255)
    # print(B_channel / (len(train_datas) * 32 * 32) * 255)
    # 预处理 去均值
    # for i in range(3):
    #     # test_datas[:, :, i] -= cifar_mean[0]
    #     train_datas[:, :, i] -= cifar_mean[0]
    # train('cifar_test_train.hdf5', train_datas, train_labels, test_datas, test_labels, batch_size=128, epochs=100)
    # print(cifar_mean)
    pass
