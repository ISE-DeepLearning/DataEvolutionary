# -*- coding=utf-8 -*-

from image_process import image_util as util
import numpy as np
import scipy.misc
import os
import random
import math

# generate the split and join image data
# generate the mix up image data

original_data_path = '../mnist/training_npy/'
center_data_save_path = '../mnist/center/'
mix_up_save_path = '../mnist/mix_up/'


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
        npy_dir_path = os.path.join(center_data_save_path, 'npy/')
        if not os.path.exists(npy_dir_path):
            os.makedirs(npy_dir_path)
        np.save(os.path.join(npy_dir_path, str(i) + '.npy'), augment_datas[i])


def generate_horizontal_samples():
    horizontal_image_path = '../mnist/horizontal/img'
    horizontal_npy_path = '../mnist/horizontal/npy'
    for i in range(10):
        npy_dir_path = os.path.join(center_data_save_path, 'npy/')
        path = os.path.join(npy_dir_path, str(i) + '.npy')
        datas = np.load(path)
        tops = [util.cut2part(data.reshape(28, 28), 'horizontal')[0] for data in datas]
        bottoms = [util.cut2part(data.reshape(28, 28), 'horizontal')[1] for data in datas]
        horizontal_samples = [util.join2part(random.choice(tops), random.choice(bottoms), 'vertical').flatten() for
                              index in range(len(datas))]
        for j in range(len(horizontal_samples)):
            print(str(i) + '_' + str(j))
            horizontal_sample = horizontal_samples[j]
            image_dir_path = os.path.join(horizontal_image_path, str(i))
            if not os.path.exists(image_dir_path):
                os.makedirs(image_dir_path)
            scipy.misc.toimage(horizontal_sample.reshape((28, 28)), cmin=0.0, cmax=1.0).save(
                os.path.join(image_dir_path, str(i) + '_' + str(j) + '.png'))
        if not os.path.exists(horizontal_npy_path):
            os.makedirs(horizontal_npy_path)
        np.save(os.path.join(horizontal_npy_path, str(i) + '.npy'), horizontal_samples)


def generate_vertical_samples():
    vertical_image_path = '../mnist/vertical/img'
    vertical_npy_path = '../mnist/vertical/npy'
    for i in range(10):
        npy_dir_path = os.path.join(center_data_save_path, 'npy/')
        path = os.path.join(npy_dir_path, str(i) + '.npy')
        datas = np.load(path)
        left = [util.cut2part(data.reshape(28, 28), 'vertical')[0] for data in datas]
        right = [util.cut2part(data.reshape(28, 28), 'vertical')[1] for data in datas]
        vertical_samples = [util.join2part(random.choice(left), random.choice(right), 'horizontal').flatten() for
                            index in range(len(datas))]
        for j in range(len(vertical_samples)):
            print(str(i) + '_' + str(j))
            vertical_sample = vertical_samples[j]
            image_dir_path = os.path.join(vertical_image_path, str(i))
            if not os.path.exists(image_dir_path):
                os.makedirs(image_dir_path)
            scipy.misc.toimage(vertical_sample.reshape((28, 28)), cmin=0.0, cmax=1.0).save(
                os.path.join(image_dir_path, str(i) + '_' + str(j) + '.png'))
        if not os.path.exists(vertical_npy_path):
            os.makedirs(vertical_npy_path)
        np.save(os.path.join(vertical_npy_path, str(i) + '.npy'), vertical_samples)


def generate_4_part_samples():
    cross_image_path = '../mnist/cross/img'
    cross_npy_path = '../mnist/cross/npy'
    for i in range(10):
        npy_dir_path = os.path.join(center_data_save_path, 'npy/')
        path = os.path.join(npy_dir_path, str(i) + '.npy')
        datas = np.load(path)
        nw = [util.cut4part(data.reshape(28, 28))[0] for data in datas]
        ne = [util.cut4part(data.reshape(28, 28))[1] for data in datas]
        sw = [util.cut4part(data.reshape(28, 28))[2] for data in datas]
        se = [util.cut4part(data.reshape(28, 28))[3] for data in datas]
        cross_samples = [
            util.join4part(random.choice(nw), random.choice(ne), random.choice(sw), random.choice(se)).flatten() for
            index in range(len(datas))]
        for j in range(len(cross_samples)):
            print(str(i) + '_' + str(j))
            cross_sample = cross_samples[j]
            image_dir_path = os.path.join(cross_image_path, str(i))
            if not os.path.exists(image_dir_path):
                os.makedirs(image_dir_path)
            scipy.misc.toimage(cross_sample.reshape((28, 28)), cmin=0.0, cmax=1.0).save(
                os.path.join(image_dir_path, str(i) + '_' + str(j) + '.png'))
        if not os.path.exists(cross_npy_path):
            os.makedirs(cross_npy_path)
        np.save(os.path.join(cross_npy_path, str(i) + '.npy'), cross_samples)


def generate_mix_up_samples(mode='max'):
    for i in range(10):
        # n*784 data
        original_datas = np.load(os.path.join(original_data_path, str(i) + '.npy'))
        # n centers
        center_datas = [util.find_center(data, shape=(28, 28)) for data in original_datas]
        # n angels
        angle_datas = [util.cal_angle(data, shape=(28, 28)) for data in original_datas]
        # generate npy save path
        mode_npy_save_path = os.path.join(mix_up_save_path, mode + '/npy')
        if not os.path.exists(mode_npy_save_path):
            os.makedirs(mode_npy_save_path)
        # generate image save path
        mode_image_save_path = os.path.join(mix_up_save_path, mode + '/img/' + str(i))
        if not os.path.exists(mode_image_save_path):
            os.makedirs(mode_image_save_path)
        datas = []
        for j in range(len(original_datas)):
            print(str(i) + '_' + str(j))
            index1 = random.randint(0, len(original_datas) - 1)
            index2 = random.randint(0, len(original_datas) - 1)
            data1 = original_datas[index1]
            data2 = original_datas[index2]
            center1 = center_datas[index1]
            center2 = center_datas[index2]
            angle1 = angle_datas[index1]
            angle2 = angle_datas[index2]
            change_angle = math.fabs(angle1 - angle2)
            if angle1 > angle2:
                data2 = util.rotate(data2, change_angle, fill=0, shape=(28, 28))
            else:
                data1 = util.rotate(data1, change_angle, fill=0, shape=(28, 28))
            # data1 center move to data2 center
            data1 = util.move(data1, x=center2[0] - center1[0], y=center2[1] - center1[1], shape=(28, 28))
            # mix up two random image data
            mix_up_data = util.mix(data1, data2, mode=mode, shape=(28, 28))
            scipy.misc.toimage(mix_up_data.reshape((28, 28)), cmin=0.0, cmax=1.0).save(
                os.path.join(mode_image_save_path, str(i) + '_' + str(j) + '.png'))
            datas.append(mix_up_data.flatten())
        np.save(os.path.join(mode_npy_save_path, str(i) + '.npy'), datas)


if __name__ == '__main__':
    # generate_center_image()
    generate_vertical_samples()
    generate_horizontal_samples()
    generate_4_part_samples()
    # generate_mix_up_samples('max')
    # generate_mix_up_samples('min')
    # generate_mix_up_samples('average')
    # generate_mix_up_samples('add')
    pass
