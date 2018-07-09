# -*- coding: utf-8 -*-
from keras import Model, Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import json
import os
import data_query as dq
import mnist.data_process.config as config

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
                        epochs=10):
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
    for time in range(10):
        result = []
        model_path = os.path.join('../experiment', config.exp_index, 'model', str(time))
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # repeat
        train_data, train_label = dq.repeat_data()
        print(np.shape(train_data))
        print(np.shape(train_label))
        model_train_process(
            os.path.join(model_path, 'repeat_samples_model.hdf5'),
            train_data,
            train_label,
            test_images=mnist.test.images,
            test_labels=mnist.test.labels)

        # evolution
        train_data, train_label = dq.evolution_data()
        print(np.shape(train_data))
        print(np.shape(train_label))
        model_train_process(
            os.path.join(model_path, 'evolution_samples_model.hdf5'),
            train_data,
            train_label,
            test_images=mnist.test.images,
            test_labels=mnist.test.labels)

        # save data
        with open(
                # 'dnn_result_' + str(sample_count[i]) + '_' + str(original_rates[i]) + '_' + str(rates[i]) + '.json',
                os.path.join('../experiment', config.exp_index, 'dnn_result_' + str(time) + '.json'),
                'w') as outfile:
            json.dump(result, outfile, ensure_ascii=False)
            outfile.write('\n')
