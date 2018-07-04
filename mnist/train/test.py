# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
data1 = mnist.train.images[0]
datas = mnist.train.images.reshape((-1, 28, 28, 1))
data2 = datas[0].flatten()
print(data1 == data2)
