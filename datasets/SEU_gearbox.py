import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


sample_len = 1024
label = [i for i in range(0, 4)]
folder = './data/SEU gearbox/gearset'
dataset_dir = [
    os.path.join('{}/Health_30_2.csv'.format(folder)),
    os.path.join('{}/Miss_30_2.csv'.format(folder)),
    os.path.join('{}/Root_30_2.csv'.format(folder)),
    os.path.join('{}/Chipped_30_2.csv'.format(folder)),
    # os.path.join('{}/Surface_30_2.csv'.format(folder))
    ]


# save dataset
def SEU_gearbox_save(args):
    data_normal = []
    lab_normal = []
    data_fault = []
    lab_fault = []

    for i in tqdm(label):
        data, lab = data_load(args, dataset_dir[i], label=label[i])
        if label[i] == 0:
            data_normal = data
            lab_normal = lab
        else:
            data_fault += data
            lab_fault += lab

    # creat the saving file
    if not os.path.exists('./data/save_dataset'):
        os.makedirs('./data/save_dataset')
    list_data = [data_normal, lab_normal, data_fault, lab_fault]
    np.save('./data/save_dataset/SEU_gearbox.npy', list_data)


# load data from the file
def data_load(args, root, label):
    data_all = []
    data = []
    lab = []

    for i in [0, 1]:  # two sensor channels
        fl = pd.read_csv(root, sep='\t', usecols=[i], header=None, skiprows=100)
        fl = fl.values
        fl = fl.reshape(-1, )
        data_all.append(fl)
    data_org = np.array(data_all, dtype=np.float32)  # all sensors with all original data

    start, end = 0, sample_len
    for j in range(args.target_train+args.test_num):
        x1 = data_org[0, start:end]
        x2 = data_org[1, start:end]

        if args.signal_type == 'FD':
            x1 = np.fft.fft(x1)
            x1 = np.abs(x1) / len(x1)
            x1 = x1[range(int(x1.shape[0] / 2))]
            x2 = np.fft.fft(x2)
            x2 = np.abs(x2) / len(x2)
            x2 = x2[range(int(x2.shape[0] / 2))]

        # Normalization
        x1 = (x1 - x1.min()) / (x1.max() - x1.min())
        x2 = (x2 - x2.min()) / (x2.max() - x2.min())

        sample = np.concatenate((x1, x2), axis=0).reshape(2, -1).astype(np.float32)
        data.append(sample)
        lab.append(label)
        start += 1200
        end += 1200

    return data, lab
