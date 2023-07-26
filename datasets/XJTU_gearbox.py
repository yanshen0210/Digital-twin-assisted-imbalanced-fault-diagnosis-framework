import os
from tqdm import tqdm
import numpy as np
import pandas as pd


sample_len = 1024
label = [i for i in range(0, 4)]
folder = './data/XJTU_Gearbox'
dataset_dir = [
    os.path.join('{}/2ndPlanetary_normalstate'.format(folder)),
    os.path.join('{}/2ndPlanetary_missingtooth'.format(folder)),
    os.path.join('{}/2ndPlanetary_rootcracks'.format(folder)),
    os.path.join('{}/2ndPlanetary_brokentooth'.format(folder)),
    # os.path.join('{}/2ndPlanetary_toothwear'.format(folder))
    ]


# save dataset
def XJTU_gearbox_save(args):
    data_normal = []
    lab_normal = []
    data_fault = []
    lab_fault = []
    channel = os.listdir(os.path.join(dataset_dir[0]))

    for i in tqdm(label):
        # two sensor channels
        channel1 = os.path.join(dataset_dir[i], channel[0])
        channel2 = os.path.join(dataset_dir[i], channel[1])

        data, lab = data_load(args, channel1, channel2, label=label[i])
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
    np.save('./data/save_dataset/XJTU_gearbox.npy', list_data)


# load data from the file
def data_load(args, root1, root2, label):
    data = []
    lab = []

    fl = pd.read_csv(root1, sep='\t',  header=None, skiprows=100)
    fl = fl.values
    fl = np.array(fl, dtype=np.float32).reshape(-1)

    f2 = pd.read_csv(root2, sep='\t',  header=None, skiprows=100)
    f2 = f2.values
    f2 = np.array(f2, dtype=np.float32).reshape(-1)

    start, end = 0, sample_len
    for j in range(args.target_train+args.test_num):
        x1 = fl[start:end]
        x2 = f2[start:end]

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
        start += 1500
        end += 1500

    return data, lab
