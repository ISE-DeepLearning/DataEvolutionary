# -*- coding: utf-8 -*-
from keras import Model, Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import json
import os
import data_query as dq

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''
3 layers fully connected nueral net work
784-128-64-10
params [model_save_path,train_samples,train_labels,batch_size,epochs]
'''

# print(np.shape(mnist.train.images))
# print(np.shape(mnist.train.labels))
# print(np.shape(mnist.test.images))
# print(np.shape(mnist.test.labels))

result = []


def model_train_process(model_save_path, train_images, train_labels, test_images, test_labels, batch_size=256,
                        epochs=5):
    input_data = Input((28 * 28,))
    temp_data1 = Dense(128)(input_data)
    temp_data2 = Activation('relu')(temp_data1)
    temp_data3 = Dense(64)(temp_data2)
    temp_data4 = Activation('relu')(temp_data3)
    temp_data5 = Dense(10)(temp_data4)
    output_data = Activation('softmax')(temp_data5)
    model = Model(inputs=[input_data], outputs=[output_data])
    modelcheck = ModelCheckpoint(model_save_path, monitor='loss', verbose=1, save_best_only=True)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    # significant params [batch_size,epochs]
    # we can build our own train images and labels
    model.fit([train_images], [train_labels], batch_size=batch_size, epochs=epochs, callbacks=[modelcheck],
              validation_data=(test_images, test_labels))

    score = model.evaluate(test_images, test_labels, verbose=0)
    print('测试集 score(val_loss): %.4f' % score[0])
    print('测试集 accuracy: %.4f' % score[1])
    result.append({'path': model_save_path, 'val_loss': score[0], 'acc': score[1]})
    print("train DNN done, save at " + model_save_path)


def retrain_process(model_path, train_images, train_labels, test_images, test_labels, batch_size=256, epochs=5):
    pass


if __name__ == '__main__':
    # 200/500/1000/2000/3000/5000/7000/10000
    rates = [25 / 55000, 50 / 55000, 100 / 55000, 250 / 55000, 500 / 55000, 1000 / 55000, 1500 / 55000, 2500 / 55000,
             3500 / 55000,
             5000 / 55000]
    original_rates = [25 / 55000, 50 / 55000, 100 / 55000, 250 / 55000, 500 / 55000, 1000 / 55000, 1500 / 55000,
                      2500 / 55000,
                      3500 / 55000,
                      5000 / 55000]
    sample_count = [50, 100, 200, 500, 1000, 2000, 3000, 5000, 7000, 10000]
    # rates = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # original_rates = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # time = 'just_an_idea'
    # for time in range(1, 10):
    for time in range(10):
        for i in range(len(rates)):
            result = []
            # if not os.path.exists(
            #         './dnn_models/' + str(sample_count[i]) + '_' + str(original_rates[i]) + '_' + str(rates[i])):
            #     os.makedirs('./dnn_models/' + str(sample_count[i]) + '_' + str(original_rates[i]) + '_' + str(rates[i]))
            if not os.path.exists('./dnn_models/' + str(sample_count[i]) + '_' + str(time)):
                os.makedirs('./dnn_models/' + str(sample_count[i]) + '_' + str(time))

            # double
            train_data, train_label = dq.double_original_training_data(original_rates[i], rates[i])
            print(np.shape(train_data))
            print(np.shape(train_label))
            model_train_process(
                # './dnn_models/' + str(sample_count[i]) + '_' + str(original_rates[i]) + '_' + str(
                #     rates[i]) + '/double_samples_model.hdf5',
                './dnn_models/' + str(sample_count[i]) + '_' + str(time) + '/double_samples_model.hdf5',
                train_data,
                train_label,
                test_images=mnist.test.images,
                test_labels=mnist.test.labels)

            # mix max
            train_data, train_label = dq.original_and_mix_max_data(original_rates[i], rates[i])
            print(np.shape(train_data))
            print(np.shape(train_label))
            model_train_process(
                # './dnn_models/' + str(sample_count[i]) + '_' + str(original_rates[i]) + '_' + str(
                #     rates[i]) + '/mix_max_model.hdf5',
                './dnn_models/' + str(sample_count[i]) + '_' + str(time) + '/mix_max_model.hdf5',
                train_data, train_label,
                test_images=mnist.test.images,
                test_labels=mnist.test.labels)

            # mix min
            train_data, train_label = dq.original_and_mix_min_data(original_rates[i], rates[i])
            print(np.shape(train_data))
            model_train_process(
                # './dnn_models/' + str(sample_count[i]) + '_' + str(original_rates[i]) + '_' + str(
                #     rates[i]) + '/mix_min_model.hdf5',
                './dnn_models/' + str(sample_count[i]) + '_' + str(time) + '/mix_min_model.hdf5',
                train_data, train_label,
                test_images=mnist.test.images,
                test_labels=mnist.test.labels)

            # mix average
            train_data, train_label = dq.original_and_mix_average_data(original_rates[i], rates[i])
            print(np.shape(train_data))
            print(np.shape(train_label))
            model_train_process(
                # './dnn_models/' + str(sample_count[i]) + '_' + str(original_rates[i]) + '_' + str(
                #     rates[i]) + '/mix_average_model.hdf5',
                './dnn_models/' + str(sample_count[i]) + '_' + str(time) + '/mix_average_model.hdf5',
                train_data, train_label,
                test_images=mnist.test.images,
                test_labels=mnist.test.labels)

            # mix add
            train_data, train_label = dq.original_and_mix_add_data(original_rates[i], rates[i])
            print(np.shape(train_data))
            print(np.shape(train_label))
            model_train_process(
                # './dnn_models/' + str(sample_count[i]) + '_' + str(original_rates[i]) + '_' + str(
                # rates[i]) + '/mix_add_model.hdf5',
                './dnn_models/' + str(sample_count[i]) + '_' + str(time) + '/mix_add_model.hdf5',
                train_data, train_label,
                test_images=mnist.test.images,
                test_labels=mnist.test.labels)

            # horizontal
            train_data, train_label = dq.original_and_horizontal_data(original_rates[i], rates[i])
            print(np.shape(train_data))
            print(np.shape(train_label))
            model_train_process(
                # './dnn_models/' + str(sample_count[i]) + '_' + str(original_rates[i]) + '_' + str(
                #     rates[i]) + '/horizontal_model.hdf5',
                './dnn_models/' + str(sample_count[i]) + '_' + str(time) + '/horizontal_model.hdf5',
                train_data, train_label,
                test_images=mnist.test.images,
                test_labels=mnist.test.labels)

            # vertical
            train_data, train_label = dq.original_and_vertical_data(original_rates[i], rates[i])
            print(np.shape(train_data))
            print(np.shape(train_label))
            model_train_process(
                # './dnn_models/' + str(sample_count[i]) + '_' + str(original_rates[i]) + '_' + str(
                #     rates[i]) + '/vertical_model.hdf5',
                './dnn_models/' + str(sample_count[i]) + '_' + str(time) + '/vertical_model.hdf5',
                train_data, train_label,
                test_images=mnist.test.images,
                test_labels=mnist.test.labels)

            # mix max
            train_data, train_label = dq.original_and_cross_data(original_rates[i], rates[i])
            print(np.shape(train_data))
            print(np.shape(train_label))
            model_train_process(
                # './dnn_models/' + str(sample_count[i]) + '_' + str(original_rates[i]) + '_' + str(
                #     rates[i]) + '/cross_model.hdf5',
                './dnn_models/' + str(sample_count[i]) + '_' + str(time) + '/cross_model.hdf5',
                train_data, train_label,
                test_images=mnist.test.images,
                test_labels=mnist.test.labels)
            # save data
            with open(
                    # 'dnn_result_' + str(sample_count[i]) + '_' + str(original_rates[i]) + '_' + str(rates[i]) + '.json',
                    'dnn_result_' + str(sample_count[i]) + '_' + str(time) + '.json',
                    'w') as outfile:
                json.dump(result, outfile, ensure_ascii=False)
                outfile.write('\n')
