# -*- coding=utf-8 -*-
datasets = {
    'mnist': {
        'origin_dir_path': './mnist/MNIST_data/',
        'training_dir_path': './mnist/training_npy/',
        'test_dir_path': './mnist/test_npy/'
    },
    'cifar-10': {
        'origin_dir_path': './cifar-10/cifar-10-batches-py/',
        'training_dir_path': './cifar-10/training_npy/',
        'test_dir_path': './cifar-10/test_npy/'
    }
}

exp = {
    'exp_dir_path': './exp/',
    'exp_index': 'exp2',
    'dataset': 'cifar-10',
    'shape': (3, 32, 32),
    'label_num': 10,
    'sample_every_label': 10,
    'upper': 255,
    'mode': ['all'],
    'repeat_times': 5
}
