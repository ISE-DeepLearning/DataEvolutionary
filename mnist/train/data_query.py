# -*- coding:utf-8 -*-
import os
import numpy as np

# original data
original_data_path = '../training_npy/'
vertical_data_path = '../vertical/npy/'
horizontal_data_path = '../horizontal/npy/'
cross_data_path = '../cross/npy/'
mix_max_data_path = '../mix_up/max/npy/'
mix_min_data_path = '../mix_up/min/npy/'
mix_average_data_path = '../mix_up/average/npy/'
mix_add_data_path = '../mix_up/add/npy/'
original_test_path = '../test_npy/'


def original_test_data():
    images = []
    labels = []
    for i in range(10):
        npy_path = os.path.join(original_test_path, str(i) + '.npy')
        image_set = np.load(npy_path)
        label_set = np.zeros(shape=(len(image_set), 10))
        label_set[:, i] = 1
        images.extend(image_set)
        labels.extend(label_set)
    # images = np.row_stack((images, images))
    # labels = np.row_stack((labels, labels))
    return images, labels


def double_original_training_data():
    images = []
    labels = []
    for i in range(10):
        npy_path = os.path.join(original_data_path, str(i) + '.npy')
        image_set = np.load(npy_path)
        label_set = np.zeros(shape=(len(image_set), 10))
        label_set[:, i] = 1
        images.extend(image_set)
        labels.extend(label_set)
    images = np.row_stack((images, images))
    labels = np.row_stack((labels, labels))
    return images, labels


def original_and_vertical_data():
    pass


def original_and_horizontal_data():
    pass


def original_and_cross_data():
    pass


def original_and_mix_max_data():
    images = []
    labels = []
    for i in range(10):
        original_npy_path = os.path.join(original_data_path, str(i) + '.npy')
        mix_npy_path = os.path.join(mix_max_data_path, str(i) + '.npy')
        original_image_set = np.load(original_npy_path)
        mix_image_set = np.load(mix_npy_path)
        label_set = np.zeros(shape=(len(original_image_set) + len(mix_image_set), 10))
        label_set[:, i] = 1
        images.extend(original_image_set)
        images.extend(mix_image_set)
        labels.extend(label_set)
    return images, labels


def original_and_mix_min_data():
    images = []
    labels = []
    for i in range(10):
        original_npy_path = os.path.join(original_data_path, str(i) + '.npy')
        mix_npy_path = os.path.join(mix_min_data_path, str(i) + '.npy')
        original_image_set = np.load(original_npy_path)
        mix_image_set = np.load(mix_npy_path)
        label_set = np.zeros(shape=(len(original_image_set) + len(mix_image_set), 10))
        label_set[:, i] = 1
        images.extend(original_image_set)
        images.extend(mix_image_set)
        labels.extend(label_set)
    return images, labels


def original_and_mix_average_data():
    images = []
    labels = []
    for i in range(10):
        original_npy_path = os.path.join(original_data_path, str(i) + '.npy')
        mix_npy_path = os.path.join(mix_average_data_path, str(i) + '.npy')
        original_image_set = np.load(original_npy_path)
        mix_image_set = np.load(mix_npy_path)
        label_set = np.zeros(shape=(len(original_image_set) + len(mix_image_set), 10))
        label_set[:, i] = 1
        images.extend(original_image_set)
        images.extend(mix_image_set)
        labels.extend(label_set)
    return images, labels


def original_and_mix_add_data():
    images = []
    labels = []
    for i in range(10):
        original_npy_path = os.path.join(original_data_path, str(i) + '.npy')
        mix_npy_path = os.path.join(mix_add_data_path, str(i) + '.npy')
        original_image_set = np.load(original_npy_path)
        mix_image_set = np.load(mix_npy_path)
        label_set = np.zeros(shape=(len(original_image_set) + len(mix_image_set), 10))
        label_set[:, i] = 1
        images.extend(original_image_set)
        images.extend(mix_image_set)
        labels.extend(label_set)
    return images, labels


def original_and_vertical_data():
    images = []
    labels = []
    for i in range(10):
        original_npy_path = os.path.join(original_data_path, str(i) + '.npy')
        vertical_npy_path = os.path.join(vertical_data_path, str(i) + '.npy')
        original_image_set = np.load(original_npy_path)
        vertical_image_set = np.load(vertical_npy_path)
        label_set = np.zeros(shape=(len(original_image_set) + len(vertical_image_set), 10))
        label_set[:, i] = 1
        images.extend(original_image_set)
        images.extend(vertical_image_set)
        labels.extend(label_set)
    return images, labels


def original_and_horizontal_data():
    images = []
    labels = []
    for i in range(10):
        original_npy_path = os.path.join(original_data_path, str(i) + '.npy')
        horizontal_npy_path = os.path.join(horizontal_data_path, str(i) + '.npy')
        original_image_set = np.load(original_npy_path)
        horizontal_image_set = np.load(horizontal_npy_path)
        label_set = np.zeros(shape=(len(original_image_set) + len(horizontal_image_set), 10))
        label_set[:, i] = 1
        images.extend(original_image_set)
        images.extend(horizontal_image_set)
        labels.extend(label_set)
    return images, labels


def original_and_cross_data():
    images = []
    labels = []
    for i in range(10):
        original_npy_path = os.path.join(original_data_path, str(i) + '.npy')
        cross_npy_path = os.path.join(vertical_data_path, str(i) + '.npy')
        original_image_set = np.load(original_npy_path)
        cross_image_set = np.load(cross_npy_path)
        label_set = np.zeros(shape=(len(original_image_set) + len(cross_image_set), 10))
        label_set[:, i] = 1
        images.extend(original_image_set)
        images.extend(cross_image_set)
        labels.extend(label_set)
    return images, labels
