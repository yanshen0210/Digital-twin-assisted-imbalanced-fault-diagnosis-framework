import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


# load dataset
class dataset(Dataset):
    def __init__(self, list_data):
        self.data = list_data['data'].tolist()
        self.labels = list_data['label'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        label = self.labels[item]
        return sample, label


class ADAMS_XJTU(object):
    num_sensor = 2
    num_classes = 4

    def data_split(self, args, op_num):
        # source data
        list_source = np.load('./data/save_dataset/ADAMS_XJTU.npy', allow_pickle=True)
        source_train = pd.DataFrame({"data": list_source[0], "label": list_source[1]})

        # target data
        list_target = np.load('./data/save_dataset/XJTU_gearbox.npy', allow_pickle=True)
        target_normal = pd.DataFrame({"data": list_target[0], "label": list_target[1]})
        target_fault = pd.DataFrame({"data": list_target[2], "label": list_target[3]})

        num_normal = args.target_train - args.target_train_fault * 3
        target_train, target_normal = train_test_split(target_normal, train_size=num_normal, random_state=op_num)
        _, target_test = train_test_split(target_normal, test_size=args.test_num, random_state=op_num)
        target_fault, target_fault_test = train_test_split(target_fault, test_size=args.test_num * 3,
                                                           random_state=op_num, stratify=target_fault["label"])
        target_fault_train, _ = train_test_split(target_fault, train_size=args.target_train_fault * 3,
                                                 random_state=op_num, stratify=target_fault["label"])

        target_train = target_train.values.T.tolist()
        target_test = target_test.values.T.tolist()
        target_fault_train = target_fault_train.values.T.tolist()
        target_fault_test = target_fault_test.values.T.tolist()

        for i in range(2):
            target_train[i] += target_fault_train[i]
            target_test[i] += target_fault_test[i]
        target_train = pd.DataFrame({"data": target_train[0], "label": target_train[1]})
        target_test = pd.DataFrame({"data": target_test[0], "label": target_test[1]})

        source_train = dataset(source_train)
        target_train = dataset(target_train)
        target_test = dataset(target_test)

        return source_train, target_train, target_test
