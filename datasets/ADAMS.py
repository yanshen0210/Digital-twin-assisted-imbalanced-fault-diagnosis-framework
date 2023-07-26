import os
from tqdm import tqdm
import numpy as np
import pandas as pd


sample_len = 1024
label = [3, 2, 1]
dataset_dir = [
    os.path.join('./data/行星齿轮箱 有限元 仿真数据/y轴'),
    os.path.join('./data/行星齿轮箱 有限元 仿真数据/z轴')
    ]

# save dataset
def ADAMS_save(args):
    data1 = []
    lab1 = []
    fault_mode = os.listdir(os.path.join(dataset_dir[0]))

    for i in tqdm(range(len(label))):
        # two sensor channels
        channel1 = os.path.join(dataset_dir[0], fault_mode[i])
        channel2 = os.path.join(dataset_dir[1], fault_mode[i])

        data, lab = data_load(args, channel1, channel2, label=label[i])
        data1 += data
        lab1 += lab

    # creat the saving file
    if not os.path.exists('./data/save_dataset'):
        os.makedirs('./data/save_dataset')
    list_data = [data1, lab1]
    np.save('./data/save_dataset/{}.npy'.format(args.transfer_task), list_data)


# load data from the file
def data_load(args, root1, root2, label):
    data = []
    lab = []

    fl = pd.read_csv(root1, sep='\t', usecols=["Current   "], header=1)
    fl = fl.values
    fl = np.array(fl, dtype=np.float32).reshape(-1)[10000:]
    f2 = pd.read_csv(root2, sep='\t', usecols=["Current   "], header=1)
    f2 = f2.values
    f2 = np.array(f2, dtype=np.float32).reshape(-1)[10000:]

    start, end = 0, sample_len
    win_shift = 240000 // args.target_train

    for j in range(args.target_train):
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

        start += win_shift
        end += win_shift

    return data, lab

