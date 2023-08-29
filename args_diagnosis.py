#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
import numpy as np
import logging

from utils.logger import setlogger
from utils.train_test import train_test
from utils.train_test_base import train_test_base
from datasets.ADAMS import ADAMS_save
from datasets.XJTU_gearbox import XJTU_gearbox_save
from datasets.SEU_gearbox import SEU_gearbox_save
import gans


args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # model and data parameters
    parser.add_argument('--save_dataset', type=bool, default=False, help='whether saving the dataset')
    parser.add_argument('--transfer_task', type=str, default='ADAMS_XJTU',
                        choices=['ADAMS_SEU', 'SEU', 'ADAMS_XJTU', 'XJTU'], 
                        help='if proposed method, choosing ADAMS_SEU or ADAMS_XJTU; elif baseline, choosing SEU or XJTU')
    parser.add_argument('--signal_type', type=str, default='FD', help='TD or FD of signal ')
    parser.add_argument('--model_name', type=str, default='ResNet18', help='the name of the model')
    parser.add_argument('--test_num', type=int, default=200, help='the test number of each healthy mode')
    parser.add_argument('--target_train', type=int, default=384, help='the normal samples of target train')
    parser.add_argument('--target_train_fault', type=int, default=16, help='the fault samples of target train')
    parser.add_argument('--num_classes', type=int, default=4, help='the class of fault modes')
    parser.add_argument('--num_sensor', type=int, default=2, help='the number of sensor channel')

    # the proposed loss to overcome imbalanced
    parser.add_argument('--transfer_loss', type=str, default='SAM+MAR',
                        choices=['SAM+MAR', 'SAM', 'MAR', 'None'],
                        help='whether add the two blocks used in our paper')

    # data-level methods
    parser.add_argument('--SMOTETomek', type=bool, default=False, help='whether operate the baseline of SMOTETomek')
    parser.add_argument('--gan', type=bool, default=False, help='whether using the baseline of gan models')
    parser.add_argument('--gen_data', type=bool, default=False, help='whether generate the fault data in gan models')
    parser.add_argument("--gan_model", type=str, default='WGAN_GP', help="choosing the gan model",
                        choices=['ACGAN', 'VAE_GAN', 'WGAN_GP'])
    parser.add_argument("--gan_lr", type=float, default=0.0005, help="the learning rate of gans")
    parser.add_argument("--gan_epoch", type=int, default=3000, help="number of epochs of gan generating")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")

    # algorithm-level methods
    parser.add_argument('--cost_loss', type=bool, default=False, help='whether using the cost loss')
    parser.add_argument('--beta', type=int, default=0.99, help='the para of the CBL')
    parser.add_argument('--loss', type=str, default='CBL', help='choosing the algorithm-level method',
                        choices=['WL', 'FL', 'DWBL', 'CBL'])
    parser.add_argument('--gamma', type=int, default=2, help='the para of the FL')

    # training parameters
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--batch_size', type=int, default=128, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--bottleneck_num', type=int, default=64, help='the bottleneck layer of classier')
    parser.add_argument('--operation_num', type=int, default=5, help='the repeat operation of model')
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epoch')
    parser.add_argument('--only_test', type=bool, default=False, help='loading the trained model if only test')

    # optimization information
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--patience', type=int, default=10, help='the para of lr scheduler')
    parser.add_argument('--min_lr', type=int, default=1e-5, help='the para of lr scheduler')

    args = parser.parse_args()
    return args


args = parse_args()

if args.save_dataset:
    if args.transfer_task == 'ADAMS_XJTU':
        ADAMS_save(args)
        XJTU_gearbox_save(args)
    elif args.transfer_task == 'ADAMS_SEU':
        ADAMS_save(args)
        SEU_gearbox_save(args)

# create the result dir
save_dir = os.path.join('./results/{}_{}_{}'.format(args.transfer_task, args.target_train,
                                                    args.target_train_fault))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# set the logger
if '_' in args.transfer_task:
    setlogger(os.path.join(save_dir, args.transfer_loss + '.log'))
elif args.gan:
    setlogger(os.path.join(save_dir, args.gan_model + '.log'))
elif args.SMOTETomek:
    setlogger(os.path.join(save_dir, 'SMOTETomek' + '.log'))
else:
    setlogger(os.path.join(save_dir,  args.loss + '.log'))

# save the args
logging.info("\n")
time = datetime.strftime(datetime.now(), '%m-%d %H:%M:%S')
logging.info('{}'.format(time))
for k, v in args.__dict__.items():
    logging.info("{}: {}".format(k, v))

Accuracy = []
J = []
if '_' in args.transfer_task:
    operation = train_test(args)
    for i in range(args.operation_num):
        if args.only_test:
            operation.setup(i)
        else:
            operation.setup(i)
            operation.train(i)
        acc, j = operation.test(i)
        Accuracy.append(acc)
        J.append(j)
else:
    operation = train_test_base(args)
    for i in range(args.operation_num):
        if args.gen_data:
            gan = getattr(gans, args.gan_model)(args, i)

        if args.only_test:
            operation.setup(i)
        else:
            operation.setup(i)
            operation.train(i)
        acc, j = operation.test(i)
        Accuracy.append(acc)
        J.append(j)

Accuracy = np.array(Accuracy)*100
Accuracy_mean = Accuracy.mean()
Accuracy_var = Accuracy.var()
Accuracy_max = Accuracy.max()
Accuracy_min = Accuracy.min()
J = np.array(J)
J_mean = J.mean()
J_var = J.var()
J_max = J.max()
J_min = J.min()
Accuracy_list = ', '.join(['{:.2f}'.format(acc) for acc in Accuracy])
J_list = ', '.join(['{:.2f}'.format(j) for j in J])
logging.info('\nAll acc: {}, \nMean acc: {:.2f}, Var acc {:.2f}, Max acc {:.2f}, Min acc {:.2f}'.format(
                Accuracy_list, Accuracy_mean, Accuracy_var, Accuracy_max, Accuracy_min))
logging.info('\nAll J: {}, \nMean J: {:.2f}, Var J {:.2f}, Max J {:.2f}, Min J {:.2f}\n'.format(
    J_list, J_mean, J_var, J_max, J_min))

