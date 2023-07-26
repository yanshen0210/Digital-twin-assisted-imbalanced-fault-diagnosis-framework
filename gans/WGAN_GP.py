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
        label = self.labels[item]
        return sample, label


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        if args.signal_type == 'FD':
            self.init_size = (args.num_sensor, 1024//2)
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
            self.init_size = (args.num_sensor, 1024//2)
        else:
            self.init_size = (args.num_sensor, 1024)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.init_size)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x_flat = x.view(x.shape[0], -1)
        validity = self.model(x_flat)
        return validity


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def WGAN_GP(args, op_num):

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

    # Loss weight for gradient penalty
    lambda_gp = 20

    # Initialize generator and discriminator
    generator = Generator(args)
    discriminator = Discriminator(args)
    name = locals()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Optimizers
    for j in [1, 2, 3]:
        name['optimizer_G' + str(j)] = torch.optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
        name['optimizer_D' + str(j)] = torch.optim.Adam(discriminator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))

    # multiple of generating samples
    mul = args.target_train // args.target_train_fault
    data = np.zeros((args.target_train_fault*mul*3, args.num_sensor, 512))
    lab = np.zeros((args.target_train_fault*mul*3))
    min_loss = [100, 100, 100]
    best_epoch = [0, 0, 0]

    for epoch in range(args.gan_epoch):
        for i, (x_all, labels) in enumerate(dataloader):
            for j in sorted(set(labels.cpu().numpy())):

                # Configure input
                x = x_all[labels == j]
                x = x.repeat(mul, 1, 1)
                real_x = Variable(x.type(Tensor))

                #  ---Train Discriminator---
                name['optimizer_D' + str(j)].zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (x.shape[0], args.latent_dim))))

                # Generate a batch of x
                fake_x = generator(z)

                # Real x
                real_validity = discriminator(real_x)
                # Fake x
                fake_validity = discriminator(fake_x)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_x.data, fake_x.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

                d_loss.backward()
                name['optimizer_D' + str(j)].step()

                #  ---Train Generator---
                name['optimizer_G' + str(j)].zero_grad()

                # Generate a batch of x
                fake_x = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake x
                fake_validity = discriminator(fake_x)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                name['optimizer_G' + str(j)].step()

                if epoch % 50 == 0:
                    print('[Num-{}, Epoch-{}]  [D loss: {:.4f}]  [G loss: {:.4f}]'.format(
                        op_num, epoch, d_loss.item(), g_loss.item()))

                # save the generating data
                if epoch >= args.gan_epoch * 2 // 3 and d_loss.item() <= min_loss[j-1]:
                    min_loss[j-1] = d_loss
                    best_epoch[j-1] = epoch
                    data_fault = fake_x.data.cpu().numpy()
                    lab_fault = np.zeros(len(fake_x), dtype='int64') + j

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
    save_dir = os.path.join('./data/save_dataset/{}_WGAN_GP'.format(args.transfer_task))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    list_data = [data_list, lab_list]
    np.save('{}/{}_{}_{}.npy'.format(save_dir, args.target_train, args.target_train_fault, op_num), list_data)




