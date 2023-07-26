import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE


# load dataset
class dataset(Dataset):
    def __init__(self, list_data):
        self.data = list_data['data'].tolist()
        self.labels = list_data['label'].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = np.array(self.data[item], dtype='float32')
        label = np.array(self.labels[item], dtype='int64')
        return sample, label


class SEU(object):
    num_sensor = 2
    num_classes = 4

    def data_split(self, args, op_num):
        # target data
        list_target = np.load('./data/save_dataset/SEU_gearbox.npy', allow_pickle=True)
        target_normal = pd.DataFrame({"data": list_target[0], "label": list_target[1]})
        target_fault = pd.DataFrame({"data": list_target[2], "label": list_target[3]})

        num_normal = args.target_train - args.target_train_fault*3
        target_train, target_normal = train_test_split(target_normal, train_size=num_normal, random_state=op_num)
        _, target_test = train_test_split(target_normal, test_size=args.test_num, random_state=op_num)
        target_fault, target_fault_test = train_test_split(target_fault, test_size=args.test_num * 3,
                                                           random_state=op_num, stratify=target_fault["label"])

        if args.gan:
            list_fault_train = np.load('./data/save_dataset/SEU_{}/{}_{}_{}.npy'.format(args.gan_model,
                                                                                        args.target_train,
                                                                                        args.target_train_fault,
                                                                                        op_num), allow_pickle=True)
            target_fault_train = pd.DataFrame({"data": list_fault_train[0], "label": list_fault_train[1]})
        else:
            target_fault_train, _ = train_test_split(target_fault, train_size=args.target_train_fault * 3,
                                                     random_state=op_num, stratify=target_fault["label"])

        target_train = target_train.values.T.tolist()
        target_test = target_test.values.T.tolist()
        target_fault_train = target_fault_train.values.T.tolist()
        target_fault_test = target_fault_test.values.T.tolist()

        for i in range(2):
            target_train[i] += target_fault_train[i]
            target_test[i] += target_fault_test[i]

        if args.SMOTETomek:
            smote = SMOTE(k_neighbors=min(args.target_train_fault - 1, 5))
            smote_tomek = SMOTETomek(smote=smote)
            N = len(target_train[0])
            inputs = np.array(target_train[0]).reshape(N, -1)
            labels = np.array(target_train[1])
            inputs, labels = smote_tomek.fit_resample(inputs, labels)
            target_train[0] = inputs.reshape(len(inputs), 2, -1).tolist()
            target_train[1] = labels.tolist()

        target_train = pd.DataFrame({"data": target_train[0], "label": target_train[1]})
        target_test = pd.DataFrame({"data": target_test[0], "label": target_test[1]})

        target_train = dataset(target_train)
        target_test = dataset(target_test)

        return target_train, target_test
