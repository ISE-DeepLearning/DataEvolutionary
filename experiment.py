import os
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam

import config
import json


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
    images = np.reshape(np.array(images), (-1, shape[1], shape[2], shape[0]))
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

    images = np.reshape(np.array(images), (-1, shape[1], shape[2], shape[0]))
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
    images = np.reshape(np.array(images), (-1, shape[1], shape[2], shape[0]))
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
    result.append({'path': model_save_path, 'val_loss': score[0], 'acc': score[1]})
    model.save(model_save_path)


if __name__ == '__main__':
    for time in range(config.exp['repeat_times']):
        result = []
        test_data, test_label = get_test_data()
        model_path = os.path.join(config.exp['exp_dir_path'], config.exp['exp_index'], 'model', str(time))
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # repeat
        train_data, train_label = repeat_data()
        train(
            os.path.join(model_path, 'repeat_samples_model.hdf5'),
            train_data,
            train_label,
            test_images=test_data, test_labels=test_label)

        # evolution
        train_data, train_label = evolution_data()
        train(
            os.path.join(model_path, 'evolution_samples_model.hdf5'),
            train_data,
            train_label,
            test_images=test_data, test_labels=test_label)

        # save data
        with open(
                os.path.join(config.exp['exp_dir_path'], config.exp['exp_index'], 'cnn_result_' + str(time) + '.json'),
                'w') as outfile:
            json.dump(result, outfile, ensure_ascii=False)
            outfile.write('\n')
