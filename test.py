# -*- coding: utf-8 -*-
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.models import Model,load_model
from keras.optimizers import SGD, Adam
import pickle
import numpy as np

test = []
train = []
train_label = []
test_label = []

with open('train', 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
    print(dict.keys())
    # 0-99
    # print(dict[b'fine_labels'])
    # 0-19
    # print(dict[b'coarse_labels'])
    print(np.shape(dict[b'data']))
    train = np.reshape(dict[b'data'], (-1, 32, 32, 3))
    print(type(dict[b'fine_labels']))
    train_label = np.zeros(shape=(len(dict[b'fine_labels']), 100))
    for i in range(len(dict[b'fine_labels'])):
        classnum = dict[b'fine_labels'][i]
        train_label[i][classnum] = 1


with open('test', 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
    print(dict.keys())
    # 0-99
    # print(dict[b'fine_labels'])
    # 0-19
    # print(dict[b'coarse_labels'])
    print(np.shape(dict[b'data']))
    test = np.reshape(dict[b'data'], (-1, 32, 32, 3))
    test_label = np.zeros(shape=(len(dict[b'fine_labels']), 100))
    for i in range(len(dict[b'fine_labels'])):
        classnum = dict[b'fine_labels'][i]
        test_label[i][classnum] = 1


def train_model_process(model_save_path, train_images, train_labels, test_images, test_labels, nb_classes=10,
                        batch_size=64, epochs=3):
    input_tensor = Input((32, 32, 3))
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

    sgd = SGD(lr=0.01, decay=1e-3, momentum=0.9, nesterov=True)
    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
              validation_data=(test_images, test_labels))
    # Y_pred = model.predict(X_test, verbose=0)
    score = model.evaluate(test_images, test_labels, verbose=0)
    print('测试集 score(val_loss): %.4f' % score[0])
    print('测试集 accuracy: %.4f' % score[1])
    # result.append({'path': model_save_path, 'val_loss': score[0], 'acc': score[1]})
    model.save(model_save_path)


def retrain(model_save_path, train_images, train_labels, test_images, test_labels, nb_classes=10,
                        batch_size=64, epochs=3,save_path='another.hdf5'):
    model = load_model(model_save_path)
    model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
              validation_data=(test_images, test_labels))
    # Y_pred = model.predict(X_test, verbose=0)
    score = model.evaluate(test_images, test_labels, verbose=0)
    print('测试集 score(val_loss): %.4f' % score[0])
    print('测试集 accuracy: %.4f' % score[1])
    # result.append({'path': model_save_path, 'val_loss': score[0], 'acc': score[1]})
    model.save(save_path)

# train_model_process("train_model_for_cifar100.hdf5",train/255,train_label,test/255,test_label,nb_classes=100)
retrain("train_model_for_cifar100.hdf5",train/255,train_label,test/255,test_label,nb_classes=100)
