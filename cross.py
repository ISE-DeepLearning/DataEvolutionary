# -*- coding=utf-8 -*-
import numpy as np

'''
数据的交叉
思路：
1.样本的混合（无需模型，不保留语义）
2.样本的剪切与粘合（无需模型，不保留语义）
'''

'''
描述：按照某种模式混合两张图片，默认为平均值，上限是255
特点：无需模型，不保留语义
输入：两张图片格式一致，k*n*n的矩阵，n代表图片大小，k代表层数，通常是1或者3。模式包括average，max，min，add。cmax是图片元素的上限
输出：混合之后的图片，k*n*n的矩阵
'''


def mix_up(image1, image2, mode='average'):
    if not np.shape(image1) == np.shape(image2):
        raise Exception('two images has different shape!')
    if mode == 'average':
        result = (image1 + image2) / 2.0
    elif mode == 'max':
        result = np.maximum(image1, image2)
    elif mode == 'min':
        result = np.minimum(image1, image2)
    elif mode == 'add':
        result = image2 + image1
        result[result > 1] = 1
    else:
        result = (image1 + image2) / 2.0
    return result


'''
描述：按照某种规则剪切样本。从起始点开始，剪切一块长宽分别是w和h的部分然后交换。
特点：无需模型，不保留语义
输入：两张图片格式一致，k*n*n的矩阵，n代表图片大小，k代表层数，通常是1或者3。起始点默认（0,0），w默认none，h默认none
输出：剪切之后的图片，2张k*n*n的矩阵
'''


def exchange(image1, image2, start_point=(0, 0), w=0, h=0):
    result1, result2 = image1.copy(), image2.copy()
    if not np.shape(result1) == np.shape(result2):
        raise Exception('two images has different shape!')
    x0, y0 = start_point
    k, n = np.shape(result1)[0], np.shape(result1)[1]
    if x0 > n or y0 > n:
        raise Exception('the start point is over range!')
    if x0 + w > n:
        w = n - x0
    if y0 + h > n:
        h = n - y0
    for i in range(k):
        layer1, layer2 = result1[i], result2[i]
        part1 = layer1[y0:y0 + h][:, x0:x0 + w]
        part2 = layer2[y0:y0 + h][:, x0:x0 + w]
        swap(part1, part2)
    return result1, result2


'''
描述：矩阵元素交换
输入：两个相同大小的矩阵
'''


def swap(mat1, mat2):
    h, w = np.shape(mat1)
    for i in range(h):
        for j in range(w):
            mat1[i][j], mat2[i][j] = mat2[i][j], mat1[i][j]


if __name__ == '__main__':
    print('hello, cross!')
