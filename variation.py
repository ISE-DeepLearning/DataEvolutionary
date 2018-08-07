# -*- coding=utf-8 -*-
'''
数据的变异
思路：
1.旋转，扩大缩小，变胖变瘦，变高变矮等基础图形操作（无需模型，保留一定语义）
2.参考对抗样本生成模式（需要模型，不同的对抗样本生成模式对语义保留不同）
3.直接对图形进行语义化的修改（无需模型，保留一定语义）
'''

import cv2
import numpy as np

'''
描述：按照指定中心逆时针旋转图片，多余部分涂满指定颜色
特点：无需模型，保留一定
输入：图片，k*n*n的矩阵。角度，0到2*pi。指定的颜色,三维元组，表示rgb，默认为（0，0，0）即黑色。中心（x,y）默认为图片中心
输出：混合之后的图片，k*n*n的矩阵
'''


def rotate(image, angle, center=None):
    image = transorm_to_nnk(image)
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    # 返回旋转后的图像
    result = transorm_to_knn(rotated)
    return result

    # (h, w) = image.shape[:2]
    # if center is None:
    #     center = (w / 2, h / 2)
    # M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rotated = cv2.warpAffine(image, M, (w, h))
    # return rotated


'''
n*n*k矩阵转换成k*n*n
'''


def transorm_to_knn(image):
    if len(image.shape) == 2:
        n = image.shape[0]
        return image.reshape(1, n, n)
    else:
        n, k, temp = image.shape[0], image.shape[-1], []
        for i in range(k):
            temp.append(image[:, :, i].reshape(n, n, 1))
        return np.array(temp).reshape(k, n, n)


'''
k*n*n矩阵转换成n*n*k
'''


def transorm_to_nnk(image):
    k, n, temp = image.shape[0], image.shape[1], []
    for i in range(k):
        temp.append(image[i].reshape(n, n, 1))
    image = np.concatenate(temp, -1)
    return image


if __name__ == '__main__':
    print('hello, variation!')
