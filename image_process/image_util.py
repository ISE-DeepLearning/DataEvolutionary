# -*- coding=utf-8 -*-
import numpy as np
import math
from PIL import Image

# x represents row
# y represents column


threshold = 0
cmin = 0
cmax = 1


def change_cmin(data):
    cmin = data
    return cmin


def change_cmax(data):
    cmax = data
    return cmax


def change_threshold(data):
    threshold = data
    return threshold


def get_threshold():
    return threshold


'''
return x,y
'''


def find_center(image_data, shape=None):
    if shape is not None:
        image_data = np.reshape(image_data, shape)
    data = np.where(image_data > threshold)
    top, bottom = min(data[0]), max(data[0])
    left, right = min(data[1]), max(data[1])
    return (top + bottom + 1) // 2, (left + right + 1) // 2


'''
direction='vertical'垂直的||'horizontal'水平的
return two_image_data
'''


def cut2part(image_data, direction='horizontal', split_index=None, shape=None):
    if shape is not None:
        image_data = np.reshape(image_data, shape)
    if split_index is None:
        if direction == 'horizontal':
            split_index = (np.shape(image_data)[0] + 1) // 2
        else:
            split_index = (np.shape(image_data)[1] + 1) // 2
    return np.vsplit(image_data, (split_index,)) if direction == 'horizontal' else np.hsplit(image_data,
                                                                                             (split_index,))


'''
return arrays[(NorthWest, NorthEast, SouthWest, SouthEast)]
'''


def cut4part(image_data, x=None, y=None, shape=None):
    if shape is not None:
        image_data = np.reshape(image_data, shape)
    if x is None:
        x = (np.shape(image_data)[0] + 1) // 2
    if y is None:
        y = (np.shape(image_data)[1] + 1) // 2
    top, bottom = np.vsplit(image_data, (x,))
    return np.hsplit(top, (y,)) + np.hsplit(bottom, (y,))


'''
direction='vertical'垂直的||'horizontal'水平的
return the picture data that joined together
'''


def join2part(image1, image2, direction='horizontal'):
    return np.hstack((image1, image2)) if direction == 'horizontal' else np.vstack((image1, image2))


'''
params[NorthWest, NorthEast, SouthWest, SouthEast]
return the picture data that joined together
'''


def join4part(image_nw, image_ne, image_sw, image_se):
    return np.vstack((np.hstack((image_nw, image_ne)), np.hstack((image_sw, image_se))))


'''
get thinner image data
'''


def thin(image_data, shape=None):
    if shape is not None:
        image_data = np.reshape(image_data, shape)
    pass


'''
get fatter image data
'''


def fat(image_data, shape=None):
    if shape is not None:
        image_data = np.reshape(image_data, shape)
    pass


'''
return the angle value of the image
note: This is NOT the actual angel that the picture looks like in.
'''


def cal_angle(image_data, shape=None):
    if shape is not None:
        image_data = np.reshape(image_data, shape)
    data = np.where(image_data > threshold)
    top = min(data[0])
    top_row = image_data[top]
    top_average = top, int(np.mean(np.where(top_row > threshold)))
    bottom = max(data[0])
    bottom_row = image_data[bottom]
    bottom_average = bottom, int(np.mean(np.where(bottom_row > threshold)))
    if top_average[1] == bottom_average[1]:
        return math.pi / 2
    else:
        result = math.atan((bottom_average[0] - top_average[0]) / (top_average[1] - bottom_average[1]))
        return result if result > 0 else result + math.pi


'''
rotate angel value to the image
fill represents the pixels data that filled in (or padding)
'''


def rotate(image_data, angle, fill=0, shape=None):
    if shape is not None:
        image_data = np.reshape(image_data, shape)
    image = Image.fromarray(image_data * 255)
    im2 = image.convert('RGBA')
    out = im2.rotate((angle / math.pi) * 180)
    cover = Image.new('RGBA', out.size, (fill * 255, fill * 255, fill * 255))
    out = Image.composite(out, cover, out)
    out = out.convert('L')
    data = np.array(out.getdata()) / 255.0

    return data.reshape(np.shape(image_data))


'''
x|y can be negative or positive
x represents move up/down
y represents move left/right
fill represents the pixels data that filled in (or padding)
'''


def move(image_data, x=0, y=0, fill=0, shape=None):
    result = np.copy(image_data)
    # get the absolute x and y value
    abs_x = int(math.fabs(x))
    abs_y = int(math.fabs(y))
    if shape is not None:
        image_data = np.reshape(image_data, shape)
    # get the shape of this image
    image_shape = np.shape(image_data)
    # is this x or y out of range
    if abs_x > image_shape[0] or abs_y > image_shape[1]:
        raise Exception('x or y out of range!')
    # is this x at the border
    if abs_x != 0 and abs_x != np.shape(image_data)[0]:
        # in the area
        # x is positive
        if x > 0:
            up, bottom = cut2part(result, direction='horizontal', split_index=image_shape[0] - x, shape=image_shape)
            bottom[:] = fill
            result = join2part(bottom, up, direction='vertical')
        # x is negative
        else:
            up, bottom = cut2part(result, direction='horizontal', split_index=abs_x, shape=image_shape)
            up[:] = fill
            result = join2part(bottom, up, direction='vertical')
    # is this y at the border
    if abs_y != 0 and abs_y != np.shape(image_data)[1]:
        # in the area
        # y is positive
        if y > 0:
            left, right = cut2part(result, direction='vertical', split_index=image_shape[1] - y, shape=image_shape)
            right[:] = fill
            result = join2part(right, left, direction='horizontal')
        else:
            left, right = cut2part(result, direction='vertical', split_index=abs_y, shape=image_shape)
            left[:] = fill
            result = join2part(right, left, direction='horizontal')
    return result


'''
mode options [max,min,average,min,add,sub]
mix two images
'''


def mix(image1, image2, mode='max', shape=None):
    if shape is not None:
        image1 = np.reshape(image1, shape)
        image2 = np.reshape(image2, shape)
    result = None
    image1 = np.array(image1)
    image2 = np.array(image2)
    if not np.shape(image1) == np.shape(image2):
        raise Exception('two images has different shape!')
    if mode == 'max':
        result = np.maximum(image1, image2)
    elif mode == 'min':
        result = np.minimum(image1, image2)
    elif mode == 'average':
        result = (image1 + image2) / 2.0
    elif mode == 'add':
        result = image2 + image1
        result[result > cmax] = cmax
    elif mode == 'sub':
        result = image1 - image2
        result[result < cmin] = cmin
    else:
        raise Exception('do not have this mode! please check it or see the source code!')
    return result


if __name__ == '__main__':
    data1 = [[1, 2, 3], [4, 5, 6]]
    data2 = [[0, 3, 4], [5, 3, 2]]
    # mix_result = mix(data1, data2, mode='max')
    # mix_result = mix(data1, data2, mode='min')
    # mix_result = mix(data1, data2, mode='average')
    # mix_result = mix(data1, data2, mode='add')
    # mix_result = mix(data1, data2, mode='sub')
    # print(type(mix_result))
    # print(mix_result)
