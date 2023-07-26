import os
import numpy as np
import math
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import logging
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
        sample = np.array(self.data[item], dtype='float32')
        label = np.array(self.labels[item], dtype='int64')
        return sample, label


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        if args.signal_type == 'FD':
            self.init_size = (args.num_sensor, 1024 // 2)
        else:
            self.init_size = (args.num_sensor, 1024)

        self.fc1 = nn.Sequential(
            nn.Linear(int(np.prod(self.init_size)), 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128))
        self.fc21 = nn.Linear(128, args.latent_dim)  # mean
        self.fc22 = nn.Linear(128, args.latent_dim)  # var

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()  # e**(x/2)
        eps = torch.FloatTensor(std.size()).normal_()
        if torch.cuda.is_available():
            eps = Variable(eps.cuda())
        else:
            eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)  # 编码
        z = self.reparametrize(mu, logvar)  # 重新参数化成正态分布
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        return z, KLD  # 解码，同时输出均值方差


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        if args.signal_type == 'FD':
            self.init_size = (args.num_sensor, 1024 // 2)
        else:
            self.init_size = (args.num_sensor, 1024)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(args.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.init_size))),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.model(z)
        x = x.view(x.shape[0], *self.init_size)
        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        if args.signal_type == 'FD':
            self.init_size = (args.num_sensor, 1024 // 2)
        else:
            self.init_size = (args.num_sensor, 1024)

        self.conv = nn.Sequential(
            nn.Conv1d(args.num_sensor, 32, 3, 2, 1),  # 256
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(2, 2),  # 128
            nn.Conv1d(32, 64, 3, 2, 1),  # 64
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(2, 2),  # [B,64,32]
        )

        self.fc = nn.Sequential(
            nn.Linear(int(np.prod(self.init_size) * 64 / (args.num_sensor * 16)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

        self.f = nn.Sequential(
            nn.Conv1d(64, 1, 4, 2, 1),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = self.conv(x)
        f = self.f(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, f.squeeze()


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def lossD(scores_real, scores_fake0, scores_fake1):
    loss = 0.5 * ((scores_real - 1) ** 2).mean() + 0.5 * (scores_fake0 ** 2).mean() + 0.5 * (scores_fake1 ** 2).mean()
    return loss


def lossGD(scores_fake):
    loss = 0.5 * ((scores_fake - 1) ** 2).mean()
    return loss


def VAE_GAN(args, op_num):
    # Load the fault datasets
    if args.transfer_task == 'XJTU':
        list_target = np.load('./data/save_dataset/XJTU_gearbox.npy', allow_pickle=True)
    elif args.transfer_task == 'SEU':
        list_target = np.load('./data/save_dataset/SEU_gearbox.npy', allow_pickle=True)

    target_fault = pd.DataFrame({"data": list_target[2], "label": list_target[3]})
    target_fault, target_fault_test = train_test_split(target_fault, test_size=args.test_num * 3,
                                                       random_state=op_num, stratify=target_fault["label"])
    fault_train, _ = train_test_split(target_fault, train_size=args.target_train_fault * 3,
                                      random_state=op_num, stratify=target_fault["label"])
    fault_train = dataset(fault_train)
    dataloader = DataLoader(fault_train, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Initialize
    Enc = Encoder(args).cuda()
    Gen = Generator(args).cuda()
    Dis = Discriminator(args).cuda()
    name = locals()

    # Optimizers
    for j in [1, 2, 3]:
        name['E_trainer' + str(j)] = torch.optim.Adam(Enc.parameters(), lr=1e-3)
        name['G_trainer' + str(j)] = torch.optim.Adam(Gen.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
        name['D_trainer' + str(j)] = torch.optim.Adam(Dis.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))

    # multiple of generating samples
    mul = args.target_train // args.target_train_fault
    data = np.zeros((args.target_train_fault * mul * 3, args.num_sensor, 512))
    lab = np.zeros((args.target_train_fault * mul * 3))
    min_loss = [100, 100, 100]
    best_epoch = [0, 0, 0]

    for epoch in range(args.gan_epoch):
        for i, (x_all, labels) in enumerate(dataloader):
            for j in sorted(set(labels.cpu().numpy())):

                # Configure input
                x = x_all[labels == j]
                x = x.repeat(mul, 1, 1)
                x_r = Variable(x.type(Tensor))

                x_r0 = x_r.view(x.shape[0], -1)
                z, kld = Enc(x_r0)
                x_f = Gen(z)
                sample_noise = (torch.rand(x.shape[0], args.latent_dim) - 0.5) / 0.5  # -1 ~ 1 的均匀分布
                g_fake_seed = Variable(sample_noise).cuda()
                x_p = Gen(g_fake_seed)  # 生成的假的数据
                ld_r, _ = Dis(x_r)
                ld_f, _ = Dis(x_f)
                ld_p, _ = Dis(x_p)

                # ---------------------D training --------------------------------
                loss_D = lossD(ld_r, ld_p, ld_f)

                name['D_trainer' + str(j)].zero_grad()
                loss_D.backward()
                name['D_trainer' + str(j)].step()

                # ------------------------G & E  training------------------
                x_r0 = x_r.view(x.shape[0], -1)
                z, kld = Enc(x_r0)
                x_f = Gen(z)
                sample_noise = (torch.rand(x.shape[0], args.latent_dim) - 0.5) / 0.5  # -1 ~ 1 的均匀分布
                g_fake_seed = Variable(sample_noise).cuda()
                x_p = Gen(g_fake_seed)  # 生成的假的数据[6,3,64,64]
                ld_r, fd_r = Dis(x_r)  # [6,1]
                ld_f, fd_f = Dis(x_f)
                ld_p, _ = Dis(x_p)
                loss_GD = lossGD(ld_p)
                loss_G = 0.5 * (0.01 * (x_f - x_r).pow(2).sum() + (fd_f - fd_r).pow(2).sum()) / x.shape[0]

                name['G_trainer' + str(j)].zero_grad()
                name['E_trainer' + str(j)].zero_grad()
                kld.backward(retain_graph=True)
                (0.01 * loss_G + loss_GD).backward(torch.ones_like(loss_G))
                name['G_trainer' + str(j)].step()
                name['E_trainer' + str(j)].step()

                if epoch % 50 == 0:
                    print('[Num-{}, Epoch-{}]  [D loss: {:.4f}]  [G loss: {:.4f}]'.format(
                        op_num, epoch, loss_D.item(), loss_G.item()))

                # save the generating data
                if epoch >= args.gan_epoch * 2 // 3 and loss_D.item() <= min_loss[j - 1]:
                    min_loss[j - 1] = loss_D
                    best_epoch[j - 1] = epoch
                    data_fault = x_f.data.cpu().numpy()
                    lab_fault = np.zeros(len(x_f), dtype='int64') + j

                    data[args.target_train_fault * mul * (j - 1): args.target_train_fault * mul * j] = data_fault
                    lab[args.target_train_fault * mul * (j - 1): args.target_train_fault * mul * j] = lab_fault

    # add the real samples
    data_list = []
    lab_list = []
    data_fault = x_all.data.cpu().numpy()
    lab_fault = labels.data.cpu().numpy()

    for k in range(len(lab_fault)):
        data_list.append(data_fault[k])
        lab_list.append(lab_fault[k])

    for k in range(len(lab)):
        data_list.append(data[k])
        lab_list.append(lab[k])

    print('\nNum-{}, Best_epoch {}, D_loss {}'.format(op_num, best_epoch, min_loss))
    # creat the saving file
    save_dir = os.path.join('./data/save_dataset/{}_VAE_GAN'.format(args.transfer_task))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    list_data = [data_list, lab_list]
    np.save('{}/{}_{}_{}.npy'.format(save_dir, args.target_train, args.target_train_fault, op_num), list_data)






