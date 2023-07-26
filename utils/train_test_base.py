#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import math
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from collections import Counter

import models
import datasets
import loss


class train_test_base(object):
    def __init__(self, args):
        self.args = args

    def setup(self, op_num):
        """
        Initialize the datasets, model, loss and optimizer
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the datasets
        Dataset = getattr(datasets, args.transfer_task)
        self.datasets = {}
        self.datasets['target_train'], self.datasets['target_test'] = Dataset().data_split(args, op_num)
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x],
                                                           batch_size=(args.batch_size if x.split('_')[1] == 'train' else 100),
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           drop_last=False,
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['target_train', 'target_test']}

        # Define the model
        self.model = getattr(models, args.model_name)(in_channel=Dataset.num_sensor)
        self.classifier_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                              nn.ReLU(inplace=True), nn.Dropout(),
                                              nn.Linear(args.bottleneck_num, Dataset.num_classes)
                                              )
        # self.model_all = nn.Sequential(self.model, self.classifier_layer)

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            self.classifier_layer = torch.nn.DataParallel(self.classifier_layer)

        # Define the optimizer and learning rate decay
        self.optimizer = torch.optim.AdamW([{'params': self.model.parameters(), "lr": args.lr},
                                            {'params': self.classifier_layer.parameters(), "lr": args.lr}
                                            ])
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=args.patience,
                                                                 min_lr=args.min_lr, verbose=True)

        self.model.to(self.device)
        self.classifier_layer.to(self.device)

        # the loss to overcome unbalanced
        samples_per_cls = np.array([args.target_train, args.target_train_fault,
                                    args.target_train_fault, args.target_train_fault])
        weights = samples_per_cls.max() / samples_per_cls
        weights = torch.cuda.FloatTensor(weights)

        if args.cost_loss:
            if args.loss == 'WL':
                self.wl = nn.CrossEntropyLoss(weight=weights)
            elif args.loss == 'FL':
                self.fl = getattr(loss, 'FocalLoss')(alpha=weights)
            elif args.loss == 'DWBL':
                self.dwbl = getattr(loss, 'DWBLoss')(cls_num_list=samples_per_cls)
            elif args.loss == 'CBL':
                effective_num = 1.0 - np.power(args.beta, samples_per_cls)
                weights = (1.0 - args.beta) / np.array(effective_num)
                weights = weights / np.sum(weights) * Dataset.num_classes
                weights = torch.cuda.FloatTensor(weights)
                self.cbl = nn.CrossEntropyLoss(weight=weights)
        else:
            self.cel = nn.CrossEntropyLoss()

    def train(self, op_num):
        args = self.args
        train_time = 0
        min_loss = 100
        best_epoch = 0
        step = 0
        # train_loss = []
        # train_acc = []
        # test_loss = []
        # test_acc = []

        for epoch in range(args.max_epoch):

            # Each epoch has a training and test phase
            for phase in ['target_train', 'target_test']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0
                n = 0

                # Set model to train mode or test mode
                if phase == 'target_train':
                    self.model.train()
                    self.classifier_layer.train()
                else:
                    self.model.eval()
                    self.classifier_layer.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.set_grad_enabled(phase == 'target_train'):
                        # forward
                        features = self.model(inputs)
                        outputs = self.classifier_layer(features).squeeze()
                        if args.cost_loss:
                            if args.loss == 'WL':
                                loss = self.wl(outputs, labels)
                            elif args.loss == 'FL':
                                loss = self.fl(outputs, labels)
                            elif args.loss == 'DWBL':
                                loss = self.dwbl(outputs, labels)
                            elif args.loss == 'CBL':
                                loss = self.cbl(outputs, labels)
                        else:
                            loss = self.cel(outputs, labels)

                        pred = outputs.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item()*labels.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        n += labels.size(0)

                        # Calculate the training information
                        if phase == 'target_train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            step += 1

                epoch_loss = epoch_loss / n
                epoch_acc = epoch_acc / n

                # calculate the training time
                if phase == 'target_train':
                    train_time += time.time()-epoch_start
                    logging.info(' ')
                    self.lr_scheduler.step(epoch_loss)
                #     train_loss.append(epoch_loss)
                #     train_acc.append(epoch_acc)
                # elif phase == 'target_test':
                #     test_loss.append(epoch_loss)
                #     test_acc.append(epoch_acc)

                logging.info('Num-{}, Epoch: {}-{}, Loss: {:.4f}, Acc: {:.4f}, Time {:.4f} sec'.format(
                    op_num, epoch, phase, epoch_loss, epoch_acc, time.time()-epoch_start))

                # save the model of the min training loss
                if epoch >= args.max_epoch//2 and phase == 'target_train' and epoch_loss <= min_loss:
                    min_loss = epoch_loss
                    best_epoch = epoch
                    if args.gan:
                        # method = args.gan_model
                        save_dir = os.path.join(
                            './trained_models/{}_{}_{}/{}'.format(args.transfer_task, args.target_train,
                                                                  args.target_train_fault, args.gan_model))
                    elif args.SMOTETomek:
                        # method = 'SMOTETomek'
                        save_dir = os.path.join(
                            './trained_models/{}_{}_{}/SMOTETomek'.format(
                                args.transfer_task, args.target_train, args.target_train_fault))
                    else:
                        # method = args.loss
                        save_dir = os.path.join(
                            './trained_models/{}_{}_{}/{}'.format(args.transfer_task, args.target_train,
                                                                  args.target_train_fault, args.loss))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(self.model.state_dict(), os.path.join('{}/{}.pth'.format(
                        save_dir, 'model_operation_' + str(op_num))))
                    torch.save(self.classifier_layer.state_dict(), os.path.join('{}/{}.pth'.format(
                        save_dir, 'classifier_operation_' + str(op_num))))

        logging.info("\nTraining time {:.2f}, Best_epoch {}, Train_loss{:.4f}".format(train_time, best_epoch, min_loss))

        # mode = ['Training loss', 'Training accuracy', 'Test loss', 'Test accuracy']
        # for i in range(len(mode)):
        #     if 'SEU' in args.transfer_task:
        #         loss_acc = os.path.join('./results/SEU_512_{}/{}'.format(args.target_train_fault, mode[i]))
        #     elif 'XJTU' in args.transfer_task:
        #         loss_acc = os.path.join('./results/XJTU_384_{}/{}'.format(args.target_train_fault, mode[i]))
        #     if not os.path.exists(loss_acc):
        #         os.makedirs(loss_acc)
        #     with open("{}/{}_{}.txt".format(loss_acc, method, str(op_num)), 'w') as txt:
        #         value_dict = {0: train_loss, 1: train_acc, 2: test_loss, 3: test_acc}
        #         txt.write(str(value_dict.get(i)))

    def test(self, op_num):
        args = self.args

        # loading the best trained model
        if args.gan:
            # method = args.gan_model
            save_dir = os.path.join(
                './trained_models/{}_{}_{}/{}'.format(args.transfer_task, args.target_train,
                                                      args.target_train_fault, args.gan_model))
        elif args.SMOTETomek:
            # method = 'SMOTETomek'
            save_dir = os.path.join(
                './trained_models/{}_{}_{}/SMOTETomek'.format(
                    args.transfer_task, args.target_train, args.target_train_fault))
        else:
            # method = args.loss
            save_dir = os.path.join(
                './trained_models/{}_{}_{}/{}'.format(args.transfer_task, args.target_train,
                                                      args.target_train_fault, args.loss))

        self.model.load_state_dict(torch.load('{}/{}.pth'.format(
                save_dir, 'model_operation_' + str(op_num))), strict=False)
        self.classifier_layer.load_state_dict(torch.load('{}/{}.pth'.format(
            save_dir, 'classifier_operation_' + str(op_num))), strict=False)

        acc = 0
        loss_all = 0.0
        feature = []
        true_label = []
        pred_label = []
        self.model.eval()
        self.classifier_layer.eval()
        test_start = time.time()

        for batch_idx, (inputs, labels) in enumerate(self.dataloaders['target_test']):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # forward
            features = self.model(inputs)
            outputs = self.classifier_layer(features).squeeze()

            if args.cost_loss:
                if args.loss == 'WL':
                    loss = self.wl(outputs, labels)
                elif args.loss == 'FL':
                    loss = self.fl(outputs, labels)
                elif args.loss == 'DWBL':
                    loss = self.dwbl(outputs, labels)
                elif args.loss == 'CBL':
                    loss = self.cbl(outputs, labels)
            else:
                loss = self.cel(outputs, labels)

            pred = outputs.argmax(dim=1)
            correct = torch.eq(pred, labels).float().sum().item()
            loss_temp = loss.item()*labels.size(0)
            loss_all += loss_temp
            acc += correct
            true_label.append(labels)
            pred_label.append(pred)
            feature += [tensor.cpu().detach() for tensor in outputs]

        sample_num = len(self.dataloaders['target_test'].dataset)
        loss = loss_all / sample_num
        acc = acc / sample_num
        test_time = time.time() - test_start

        Sb, Sw, J1 = intraclass_covariance(feature, torch.stack(true_label, 0).cpu(), args.test_num, args.num_classes)
        logging.info('Num-{}, Test Loss: {:.4f}, Acc: {:.4f}, Time {:.4f} sec'.format(
            op_num, loss, acc, test_time))
        logging.info('        Sb: {:.4f}, Sw: {:.4f}, J1: {:.4f} \n'.format(Sb, Sw, J1))

        # saving the labels of prediction and reality
        # pred_label = np.array(torch.stack(pred_label, 0).cpu()).ravel()
        # true_label = np.array(torch.stack(true_label, 0).cpu()).ravel()
        # feature = np.array(torch.stack(feature, 0).cpu())
        # save_dir1 = os.path.join('./results/{}_{}_{}/Pre label/'.format(args.transfer_task, args.target_train,
        #                                                                 args.target_train_fault))
        # save_dir2 = os.path.join('./results/{}_{}_{}/True label/'.format(args.transfer_task, args.target_train,
        #                                                                  args.target_train_fault))
        # save_dir3 = os.path.join('./results/{}_{}_{}/Feature/'.format(args.transfer_task, args.target_train,
        #                                                               args.target_train_fault))
        # if not os.path.exists(save_dir1):
        #     os.makedirs(save_dir1)
        # if not os.path.exists(save_dir2):
        #     os.makedirs(save_dir2)
        # if not os.path.exists(save_dir3):
        #     os.makedirs(save_dir3)
        # np.savetxt('{}/{}_{}.txt'.format(save_dir1, method, op_num),
        #            pred_label, fmt='%.0f', newline='\n')
        # np.savetxt('{}/{}_{}.txt'.format(save_dir2, method, op_num),
        #            true_label, fmt='%.0f', newline='\n')
        # np.savetxt('{}/{}_{}.txt'.format(save_dir3, method, op_num),
        #            feature, newline='\n')
        return acc, J1


def intraclass_covariance(test_data, label, Nk, classes):
    """
    test_data:输出特征
    label:真实标签
    Nk: 每类样本的测试个数
    classes:总的类别数
    """
    mk = []
    x = [tensor.numpy() for tensor in test_data]
    y = label.numpy().reshape(Nk*classes)
    x = np.stack(x, axis=0)
    Sb, Sw = 0, 0
    for i in range(classes):
        cur_mean = np.sum(x[y == i], axis=0) / Nk
        mk.append(cur_mean)
    m = np.mean(x, axis=0)
    for j in range(classes):
        # Sb += Nk * np.dot((mk[i]-m), (mk[i]-m).T)
        Sb += Nk * np.linalg.norm((mk[j] - m))
        x_class = x[y == j]
        for k in range(Nk):
            Sw += np.linalg.norm((x_class[k]-mk[j]))

    J1 = Sb / Sw
    return Sb, Sw, J1






