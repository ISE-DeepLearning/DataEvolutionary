# -*- coding=utf-8 -*-
import numpy as np

# x represents row
# y represents column


threshold = 0

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
    pass

'''
direction='vertical'垂直的||'horizontal'水平的
return two_image_data
'''
def cut2part(image_data,direction='vertical',split_index=None,shape=None):
    if shape is not None:
        image_data = np.reshape(image_data,shape)
    if split_index is None:
        if direction == 'horizontal':
            split_index = (np.shape(image_data)[0]+1)//2
        else:
            split_index = (np.shape(image_data)[1]+1)//2
    pass

'''
return arrays[(NorthWest, NorthEast, SouthWest, SouthEast)]
'''
def cut4part(image_data,x=None,y=None,shape=None):
    if shape is not None:
        image_data = np.reshape(image_data,shape)
    if x is None:
        x = (np.shape(image_data)[0]+1)//2
    if y is None:
        y = (np.shape(image_data)[1]+1)//2
    pass

'''
direction='vertical'垂直的||'horizontal'水平的
return the picture data that joined together
'''
def join2part(image1,image2,direction='vertical'):
    pass


'''
params[NorthWest, NorthEast, SouthWest, SouthEast]
return the picture data that joined together
'''
def join4part(imageNW,imageNE,imageSW,imageSE):
    pass


'''
get thinner image data
'''
def thin(image_data,shape=None):
    if shape is not None:
        image_data = np.reshape(image_data,shape)
    pass

'''
get fatter image data
'''
def fat(image_data,shape=None):
    if shape is not None:
        image_data = np.reshape(image_data,shape)
    pass

'''
return the angel value of the image
'''
def cal_angel(image_data,shape=None):
    if shape is not None:
        image_data = np.reshape(image_data,shape)
    pass

'''
rotate angel value to the image
fill represents the pixels data that filled in (or padding)
'''
def rotate(image_data,angle,fill=0,shape=None):
    if shape is not None:
        image_data = np.reshape(image_data,shape)
    pass

'''
x|y can be negative or positive
x represents move up/down
y represents move left/right
fill represents the pixels data that filled in (or padding)
'''
def move(image_data,x=0,y=0,fill=0,shape=None):
    if shape is not None:
        image_data = np.reshape(image_data,shape)
    pass

'''
mode options [max,min,average,min,add,sub]
mix two images
'''
def mix(image1,image2,mode='max'):
    pass