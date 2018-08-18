# -*- coding=utf-8 -*-
datasets = {
    'mnist': {
        'origin_dir_path': './mnist/MNIST_data/',
        'training_dir_path': './mnist/training_npy/',
        'test_dir_path': './mnist/test_npy/'
    },
    'cifar10': {
        'origin_dir_path': './cifar10/cifar10-batches-py/',
        'training_dir_path': './cifar10/training_npy/',
        'test_dir_path': './cifar10/test_npy/'
    }
}

exp = {
    'exp_dir_path': './exp/mnist/mutation/attack',
    'exp_index': 'exp1',
    'dataset': 'mnist',
    'shape': (1, 28, 28),
    'label_num': 10,
    'sample_every_label': 30,
    # 每个label扩增的数量是多少
    # 'evolution_every_label': 100,
    'upper': 1,
    'mode': ['attack'],
    'repeat_times': 1
}

# def get_exp_configs(dataset_name, label_num, shape, upper, repeat_time):
#     exps = []
#     each_exp_samples = [5, 10, 20, 30, 50, 100]
#     for i in range(len(each_exp_samples)):
#         exp_data = exp.copy()
