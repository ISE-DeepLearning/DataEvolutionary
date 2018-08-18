import matplotlib.pyplot as plt
import json
import os
import numpy as np
import config


def draw_acc(json_path, path1, path2, classes=10):
    plt.figure
    with open(json_path, 'r') as file:
        data = json.load(file)
        epochs = list(range(classes))
        plt.plot(epochs, data['hist_original']['acc'], marker='x', label="repeat_acc")
        plt.plot(epochs, data['hist_original']['val_acc'], marker='x', label="repeat_val_acc")
        plt.plot(epochs, data['hist_evolution']['acc'], marker='*', label='evo_acc')
        plt.plot(epochs, data['hist_evolution']['val_acc'], marker='*', label="evo_val_acc")
    with open(path1, 'r') as file:
        data = json.load(file)
        epochs = list(range(classes))
        plt.plot(epochs, data['acc'], label="ori_acc")
        plt.plot(epochs, data['val_acc'], label="ori_val_acc")
    with open(path2, 'r') as file:
        data = json.load(file)
        epochs = list(range(classes))
        plt.plot(epochs, data['acc'], marker='o', label="exp_acc")
        plt.plot(epochs, data['val_acc'], marker='o', label="exp_val_acc")
    plt.legend(loc='best')
    (dirname, filename) = os.path.split(json_path)
    (filename, extension) = os.path.splitext(filename)
    plt.savefig(os.path.join(dirname, filename + '_together_acc.jpg'))
    plt.ion()
    # plt.pause(1)
    plt.close()


def draw_loss(json_path, path1, path2, classes=10):
    plt.figure
    with open(json_path, 'r') as file:
        data = json.load(file)
        epochs = list(range(classes))
        plt.plot(epochs, data['hist_original']['loss'], marker='x', label="repeat_loss")
        plt.plot(epochs, data['hist_original']['val_loss'], marker='x', label="repeat_val_loss")
        plt.plot(epochs, data['hist_evolution']['loss'], marker='*', label='evo_loss')
        plt.plot(epochs, data['hist_evolution']['val_loss'], marker='*', label="evo_val_loss")
        plt.legend(loc='upper right')
    with open(path1, 'r') as file:
        data = json.load(file)
        epochs = list(range(classes))
        plt.plot(epochs, data['acc'], label="ori_loss")
        plt.plot(epochs, data['val_acc'], label="ori_val_loss")
    with open(path2, 'r') as file:
        data = json.load(file)
        epochs = list(range(classes))
        plt.plot(epochs, data['acc'], marker='o', label="exp_loss")
        plt.plot(epochs, data['val_acc'], marker='o', label="exp_val_loss")
    (dirname, filename) = os.path.split(json_path)
    (filename, extension) = os.path.splitext(filename)
    plt.savefig(os.path.join(dirname, filename + '_together_loss.jpg'))
    plt.ion()
    # plt.pause(1)
    plt.close()


def draw_average_acc(base_path, times, classes=10):
    plt.figure
    datas_acc_ori = []
    datas_val_acc_ori = []
    datas_acc_evo = []
    datas_val_acc_evo = []
    datas_acc_repeat = []
    datas_val_acc_repeat = []
    datas_acc_exp = []
    datas_val_acc_exp = []
    for i in range(times):
        with open(os.path.join(base_path, 'cnn_result_' + str(i) + '.json'), 'r') as file:
            data = json.load(file)
            datas_acc_repeat.append(data['hist_original']['acc'])
            datas_val_acc_repeat.append(data['hist_original']['val_acc'])
            datas_acc_evo.append(data['hist_evolution']['acc'])
            datas_val_acc_evo.append(data['hist_evolution']['val_acc'])
        with open(os.path.join(base_path, 'origin_cnn_result_' + str(i) + '.json'), 'r') as file:
            data = json.load(file)
            datas_acc_ori.append(data['acc'])
            datas_val_acc_ori.append(data['val_acc'])
        with open(os.path.join(base_path, 'origin_10times_cnn_result_' + str(i) + '.json'), 'r') as file:
            data = json.load(file)
            datas_acc_exp.append(data['acc'])
            datas_val_acc_exp.append(data['val_acc'])
    epochs = list(range(classes))
    plt.plot(epochs, np.mean(datas_acc_ori, axis=0), label="ori_acc")
    plt.plot(epochs, np.mean(datas_val_acc_ori, axis=0), label="ori_val_acc")
    plt.plot(epochs, np.mean(datas_acc_evo, axis=0), marker='*', label='evo_acc')
    plt.plot(epochs, np.mean(datas_val_acc_evo, axis=0), marker='*', label="evo_val_acc")
    plt.plot(epochs, np.mean(datas_acc_repeat, axis=0), marker='x', label='repeat_acc')
    plt.plot(epochs, np.mean(datas_val_acc_repeat, axis=0), marker='x', label="repeat_val_acc")
    plt.plot(epochs, np.mean(datas_acc_exp, axis=0), marker='o', label='exp_acc')
    plt.plot(epochs, np.mean(datas_val_acc_exp, axis=0), marker='o', label="exp_val_acc")
    plt.legend(loc='best')
    plt.savefig(os.path.join(base_path, 'average_together_acc' + '.jpg'))
    plt.ion()
    # plt.pause(1)
    plt.close()


def draw_average_loss(base_path, times, classes=10):
    plt.figure
    datas_loss_ori = []
    datas_val_loss_ori = []
    datas_loss_evo = []
    datas_val_loss_evo = []
    datas_loss_repeat = []
    datas_val_loss_repeat = []
    datas_loss_exp = []
    datas_val_loss_exp = []
    for i in range(times):
        with open(os.path.join(base_path, 'cnn_result_' + str(i) + '.json'), 'r') as file:
            data = json.load(file)
            datas_loss_repeat.append(data['hist_original']['loss'])
            datas_val_loss_repeat.append(data['hist_original']['val_loss'])
            datas_loss_evo.append(data['hist_evolution']['loss'])
            datas_val_loss_evo.append(data['hist_evolution']['val_loss'])
        with open(os.path.join(base_path, 'origin_cnn_result_' + str(i) + '.json'), 'r') as file:
            data = json.load(file)
            datas_loss_ori.append(data['loss'])
            datas_val_loss_ori.append(data['val_loss'])
        with open(os.path.join(base_path, 'origin_10times_cnn_result_' + str(i) + '.json'), 'r') as file:
            data = json.load(file)
            datas_loss_exp.append(data['loss'])
            datas_val_loss_exp.append(data['val_loss'])
    epochs = list(range(classes))
    plt.plot(epochs, np.mean(datas_loss_ori, axis=0), label="ori_loss")
    plt.plot(epochs, np.mean(datas_val_loss_ori, axis=0), label="ori_val_loss")
    plt.plot(epochs, np.mean(datas_loss_evo, axis=0), marker='*', label='evo_loss')
    plt.plot(epochs, np.mean(datas_val_loss_evo, axis=0), marker='*', label="evo_val_loss")
    plt.plot(epochs, np.mean(datas_loss_repeat, axis=0), marker='x', label='repeat_loss')
    plt.plot(epochs, np.mean(datas_val_loss_repeat, axis=0), marker='x', label="repeat_val_loss")
    plt.plot(epochs, np.mean(datas_loss_exp, axis=0), marker='o', label='exp_loss')
    plt.plot(epochs, np.mean(datas_val_loss_exp, axis=0), marker='o', label="exp_val_loss")
    plt.legend(loc='best')
    plt.savefig(os.path.join(base_path, 'average_together_loss' + '.jpg'))
    plt.ion()
    # plt.pause(1)
    plt.close()


if __name__ == '__main__':
    exp_times = 6
    each_exp_times = 5
    for type_name in ['all', 'average', 'vertical', 'horizontal']:
        for i in range(1, exp_times + 1):
            for j in range(each_exp_times):
                draw_acc('./exp/mnist/' + type_name + '/exp' + str(i) + '/cnn_result_' + str(j) + '.json',
                         './exp/mnist/' + type_name + '/exp' + str(i) + '/origin_cnn_result_' + str(j) + '.json',
                         './exp/mnist/' + type_name + '/exp' + str(i) + '/origin_10times_cnn_result_' + str(j) + '.json')
                draw_loss('./exp/mnist/' + type_name + '/exp' + str(i) + '/cnn_result_' + str(j) + '.json',
                          './exp/mnist/' + type_name + '/exp' + str(i) + '/origin_cnn_result_' + str(j) + '.json',
                          './exp/mnist/' + type_name + '/exp' + str(i) + '/origin_10times_cnn_result_' + str(j) + '.json')
        for i in range(1, exp_times + 1):
            draw_average_acc('./exp/mnist/' + type_name + '/exp' + str(i) + '/', each_exp_times, 10)
            draw_average_loss('./exp/mnist/' + type_name + '/exp' + str(i) + '/', each_exp_times, 10)
