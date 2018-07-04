# -*- coding=utf-8 -*-
import image_util as util
import numpy as np
import matplotlib.pyplot as plt
import math


def load_data(number, index):
    address = '../mnist/training_npy/' + str(number) + '.npy'
    return np.load(address)[index]


def vision_cut(image_data, division=None, direction='horizontal'):
    im = image_data
    fig = plt.figure()
    if division is not None:
        if len(division) == 2:
            if direction == 'horizontal':
                left, right = division
                plotwindow = fig.add_subplot(221)
                plotwindow.imshow(im, cmap='gray')
                plotwindow = fig.add_subplot(223)
                plotwindow.imshow(left, cmap='gray')
                plotwindow = fig.add_subplot(224)
                plotwindow.imshow(right, cmap='gray')
            else:
                top, bottom = division
                plotwindow = fig.add_subplot(221)
                plotwindow.imshow(im, cmap='gray')
                plotwindow = fig.add_subplot(222)
                plotwindow.imshow(top, cmap='gray')
                plotwindow = fig.add_subplot(224)
                plotwindow.imshow(bottom, cmap='gray')
        else:
            nw, ne, sw, se = division
            plotwindow = fig.add_subplot(231)
            plotwindow.imshow(im, cmap='gray')
            plotwindow = fig.add_subplot(232)
            plotwindow.imshow(nw, cmap='gray')
            plotwindow = fig.add_subplot(233)
            plotwindow.imshow(ne, cmap='gray')
            plotwindow = fig.add_subplot(235)
            plotwindow.imshow(sw, cmap='gray')
            plotwindow = fig.add_subplot(236)
            plotwindow.imshow(se, cmap='gray')
    else:
        plotwindow = fig.add_subplot(111)
        plotwindow.imshow(im, cmap='gray')
    plt.show()


def vision_join(join, image_data, direction='horizontal'):
    im = image_data
    fig = plt.figure()
    if len(join) == 2:
        if direction == 'horizontal':
            left, right = join
            plotwindow = fig.add_subplot(221)
            plotwindow.imshow(left, cmap='gray')
            plotwindow = fig.add_subplot(223)
            plotwindow.imshow(right, cmap='gray')
            plotwindow = fig.add_subplot(224)
            plotwindow.imshow(im, cmap='gray')
        elif direction == 'vertical':
            top, bottom = join
            plotwindow = fig.add_subplot(221)
            plotwindow.imshow(top, cmap='gray')
            plotwindow = fig.add_subplot(222)
            plotwindow.imshow(bottom, cmap='gray')
            plotwindow = fig.add_subplot(224)
            plotwindow.imshow(im, cmap='gray')
    else:
        nw, ne, sw, se = join
        plotwindow = fig.add_subplot(231)
        plotwindow.imshow(nw, cmap='gray')
        plotwindow = fig.add_subplot(232)
        plotwindow.imshow(ne, cmap='gray')
        plotwindow = fig.add_subplot(234)
        plotwindow.imshow(sw, cmap='gray')
        plotwindow = fig.add_subplot(235)
        plotwindow.imshow(se, cmap='gray')
        plotwindow = fig.add_subplot(236)
        plotwindow.imshow(im, cmap='gray')
    plt.show()


def test_cut2():
    pic = load_data(3, 4).reshape(28, 28)
    div = util.cut2part(pic, 'vertical', 7)
    vision_cut(pic, div)


def test_cut4():
    pic = load_data(3, 6).reshape(28, 28)
    div = util.cut4part(pic)
    vision_cut(pic, div)


def test_join2():
    pic1 = load_data(3, 100).reshape(28, 28)
    pic2 = load_data(3, 200).reshape(28, 28)
    div1 = util.cut2part(pic1, 'vertical')
    div2 = util.cut2part(pic2, 'vertical')
    # print(np.shape(div1[0]),np.shape(div2[1]))
    join = (div1[0], div2[1])
    image = util.join2part(join[0], join[1])
    vision_join(join, image, 'vertical')


def test_join4():
    pic1 = load_data(3, 100).reshape(28, 28)
    pic2 = load_data(3, 200).reshape(28, 28)
    pic3 = load_data(3, 300).reshape(28, 28)
    pic4 = load_data(3, 400).reshape(28, 28)
    div1 = util.cut4part(pic1)
    div2 = util.cut4part(pic2)
    div3 = util.cut4part(pic3)
    div4 = util.cut4part(pic4)
    join = (div1[0], div2[1], div3[2], div4[3])
    image = util.join4part(join[0], join[1], join[2], join[3])
    vision_join(join, image)


def test_center():
    pic = load_data(8, 1).reshape(28, 28)
    print(util.find_center(pic))
    vision_cut(pic)


def test_angle():
    pic = load_data(7, 100).reshape(28, 28)
    print(util.cal_angel(pic))
    # vision_cut(pic)


def test_rotate():
    pic = load_data(7, 100).reshape(28, 28)
    test = util.rotate(pic,math.pi/8)
    vision_cut(test)


if __name__ == '__main__':
    # vision(load_data(3, 3))
    test_rotate()
