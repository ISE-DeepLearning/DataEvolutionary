# -*- coding=utf-8 -*-

from image_process import image_util as util
import numpy as np
import scipy.misc
import os

# generate the split and join image data
# generate the mix up image data

original_data_path = '../mnist/training_npy/'
center_data_save_path = '../mnist/center/'


def generate_center_image():
    augment_datas = [[], [], [], [], [], [], [], [], [], []]
    standard_center = (14, 14)
    for i in range(10):
        path = os.path.join(original_data_path, str(i) + '.npy')
        datas = np.load(path)
        # print(datas.shape)
        for j in range(len(datas)):
            print(str(i) + '_' + str(j))
            data = datas[j]
            center = util.find_center(data, shape=(28, 28))
            image_dir_path = os.path.join(center_data_save_path, 'img/' + str(i))
            if not os.path.exists(image_dir_path):
                os.makedirs(image_dir_path)
            data = util.move(data, x=standard_center[0] - center[0], y=standard_center[1] - center[1], shape=(28, 28))
            augment_datas[i].append(data.flatten())
            scipy.misc.toimage(data.reshape((28, 28)), cmin=0.0, cmax=1.0).save(
                os.path.join(image_dir_path, str(i) + '_' + str(j) + '.png'))
        npy_dir_path = os.path.join(center_data_save_path, '/npy/')
        if not os.path.exists(npy_dir_path):
            os.makedirs(npy_dir_path)
        print(np.shape(augment_datas[i]))
        np.save(os.path.join(npy_dir_path, str(i) + '.npy'), augment_datas[i])


def generate_horizontal_samples():
    pass


def generate_vertical_samples():
    pass


def generate_4_part_samples():
    pass


def generate_mix_up_samples():
    pass


if __name__ == '__main__':
    # generate_center_image()
    pass
