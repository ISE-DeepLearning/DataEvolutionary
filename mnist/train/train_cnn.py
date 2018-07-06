# -*- coding: utf-8 -*-
from keras import Model, Input

import sys
import time
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import data_query as dq
import json

mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 28*28
#
# X_train = X_train.astype('float32').reshape(-1, 28, 28, 1)
# X_test = X_test.astype('float32').reshape(-1, 28, 28, 1)
# X_train /= 255
# X_test /= 255
# print('Train:{},Test:{}'.format(len(X_train), len(X_test)))
# nb_classes = 10
# Y_train = np_utils.to_categorical(Y_train, nb_classes)
# Y_test = np_utils.to_categorical(Y_test, nb_classes)
#
# print(np.shape(X_train))  # 60000,28,28,1
# print(np.shape(Y_train))  # 60000,10
# print(np.shape(X_test))  # 10000,28,28,1
# print(np.shape(Y_test))  # 10000,10

result = []


def train_model_process(model_save_path, train_images, train_labels, test_images, test_labels, nb_classes=10,
                        batch_size=64, epochs=5):
    input_tensor = Input((28, 28, 1))
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

    temp = Dense(nb_classes)(temp)
    output = Activation('softmax')(temp)

    model = Model(input=input_tensor, outputs=output)

    model.summary()

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
              validation_data=(test_images, test_labels))
    # Y_pred = model.predict(X_test, verbose=0)
    score = model.evaluate(test_images, test_labels, verbose=0)
    print('测试集 score(val_loss): %.4f' % score[0])
    print('测试集 accuracy: %.4f' % score[1])
    result.append({'path': model_save_path, 'val_loss': score[0], 'acc': score[1]})
    model.save(model_save_path)


if __name__ == '__main__':
    # rate = 0.3
    # if not os.path.exists('./cnn_models/' + str(rate)):
    #     os.makedirs('./cnn_models/' + str(rate))
    result = []
    test_data, test_label = mnist_data.test.images, mnist_data.test.labels
    test_data = np.reshape(test_data, (-1, 28, 28, 1))

    # 200/500/1000/2000/3000/5000/7000/10000
    rates = [25 / 55000, 50 / 55000, 100 / 55000, 250 / 55000, 500 / 55000, 1000 / 55000, 1500 / 55000, 2500 / 55000,
             3500 / 55000,
             5000 / 55000]
    original_rates = [25 / 55000, 50 / 55000, 100 / 55000, 250 / 55000, 500 / 55000, 1000 / 55000, 1500 / 55000,
                      2500 / 55000,
                      3500 / 55000,
                      5000 / 55000]
    sample_count = [50, 100, 200, 500, 1000, 2000, 3000, 5000, 7000, 10000]
    for time in range(10):
        for i in range(len(rates)):
            if not os.path.exists('./cnn_models/' + str(sample_count[i]) + '_' + str(time)):
                os.makedirs('./cnn_models/' + str(sample_count[i]) + '_' + str(time))
            # double
            train_data, train_label = dq.double_original_training_data(original_rates[i], rates[i])
            train_data = np.reshape(train_data, (-1, 28, 28, 1))
            train_label = np.array(train_label)
            train_model_process('./cnn_models/' + str(sample_count[i]) + '_' + str(time) + '/double_samples_model.hdf5',
                                train_data, train_label,
                                test_images=test_data, test_labels=test_label)

            # mix max
            train_data, train_label = dq.original_and_mix_max_data(original_rates[i], rates[i])
            train_data = np.reshape(train_data, (-1, 28, 28, 1))
            train_label = np.array(train_label)

            train_model_process('./cnn_models/' + str(sample_count[i]) + '_' + str(time) + '/mix_max_model.hdf5',
                                train_data, train_label,
                                test_images=test_data, test_labels=test_label)

            # mix min
            train_data, train_label = dq.original_and_mix_min_data(original_rates[i], rates[i])
            train_data = np.reshape(train_data, (-1, 28, 28, 1))
            train_label = np.array(train_label)
            train_model_process('./cnn_models/' + str(sample_count[i]) + '_' + str(time) + '/mix_min_model.hdf5',
                                train_data, train_label,
                                test_images=test_data, test_labels=test_label)

            # mix average
            train_data, train_label = dq.original_and_mix_average_data(original_rates[i], rates[i])
            train_data = np.reshape(train_data, (-1, 28, 28, 1))
            train_label = np.array(train_label)
            train_model_process('./cnn_models/' + str(sample_count[i]) + '_' + str(time) + '/mix_average_model.hdf5',
                                train_data, train_label,
                                test_images=test_data, test_labels=test_label)

            # mix add
            train_data, train_label = dq.original_and_mix_add_data(original_rates[i], rates[i])
            train_data = np.reshape(train_data, (-1, 28, 28, 1))
            train_label = np.array(train_label)
            train_model_process('./cnn_models/' + str(sample_count[i]) + '_' + str(time) + '/mix_add_model.hdf5',
                                train_data, train_label,
                                test_images=test_data, test_labels=test_label)

            # vertical
            train_data, train_label = dq.original_and_vertical_data(original_rates[i], rates[i])
            train_data = np.reshape(train_data, (-1, 28, 28, 1))
            train_label = np.array(train_label)
            # test_data, test_label = mnist_data.test.images, mnist_data.test.labels
            # test_data = np.reshape(test_data, (-1, 28, 28, 1))
            train_model_process('./cnn_models/' + str(sample_count[i]) + '_' + str(time) + '/vertical_model.hdf5',
                                train_data, train_label,
                                test_images=test_data, test_labels=test_label)

            # horizontal
            train_data, train_label = dq.original_and_horizontal_data(original_rates[i], rates[i])
            train_data = np.reshape(train_data, (-1, 28, 28, 1))
            train_label = np.array(train_label)
            # test_data, test_label = mnist_data.test.images, mnist_data.test.labels
            # test_data = np.reshape(test_data, (-1, 28, 28, 1))
            train_model_process('./cnn_models/' + str(sample_count[i]) + '_' + str(time) + '/horizontal_model.hdf5',
                                train_data, train_label,
                                test_images=test_data, test_labels=test_label)

            # horizontal
            train_data, train_label = dq.original_and_cross_data(original_rates[i], rates[i])
            train_data = np.reshape(train_data, (-1, 28, 28, 1))
            train_label = np.array(train_label)
            # test_data, test_label = mnist_data.test.images, mnist_data.test.labels
            # test_data = np.reshape(test_data, (-1, 28, 28, 1))
            train_model_process('./cnn_models/' + str(sample_count[i]) + '_' + str(time) + '/cross_model.hdf5',
                                train_data, train_label,
                                test_images=test_data, test_labels=test_label)
            # save data
            with open('cnn_result_' + str(sample_count[i]) + '_' + str(time) + '.json', 'a') as outfile:
                json.dump(result, outfile, ensure_ascii=False)
                outfile.write('\n')
