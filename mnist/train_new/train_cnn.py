# -*- coding: utf-8 -*-
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.models import Model
from keras.optimizers import SGD, Adam
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import data_query as dq
import json
import mnist.data_process.config as config

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


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
    test_data, test_label = mnist.test.images, mnist.test.labels
    test_data = np.reshape(test_data, (-1, 28, 28, 1))
    for time in range(10):
        result = []
        model_path = os.path.join('../experiment', config.exp_index, 'model', str(time))
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # repeat
        train_data, train_label = dq.repeat_data()
        train_data = np.reshape(train_data, (-1, 28, 28, 1))
        train_label = np.array(train_label)
        print(np.shape(train_data))
        print(np.shape(train_label))
        train_model_process(
            os.path.join(model_path, 'repeat_samples_model.hdf5'),
            train_data,
            train_label,
            test_images=test_data, test_labels=test_label)

        # evolution
        train_data, train_label = dq.evolution_data()
        train_data = np.reshape(train_data, (-1, 28, 28, 1))
        train_label = np.array(train_label)
        print(np.shape(train_data))
        print(np.shape(train_label))
        train_model_process(
            os.path.join(model_path, 'evolution_samples_model.hdf5'),
            train_data,
            train_label,
            test_images=test_data, test_labels=test_label)

        # save data
        with open(
                # 'dnn_result_' + str(sample_count[i]) + '_' + str(original_rates[i]) + '_' + str(rates[i]) + '.json',
                os.path.join('../experiment', config.exp_index, 'cnn_result_' + str(time) + '.json'),
                'w') as outfile:
            json.dump(result, outfile, ensure_ascii=False)
            outfile.write('\n')
