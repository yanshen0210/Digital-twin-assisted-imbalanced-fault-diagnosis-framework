import os
import numpy as np
import math
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        if args.signal_type == 'FD':
            self.init_size = 1024 // (2*4)
        else:
            self.init_size = 1024 // 4

        self.label_emb = nn.Embedding(args.num_classes-1, args.latent_dim)
        self.l1 = nn.Sequential(nn.Linear(args.latent_dim, 128 * self.init_size))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, args.num_sensor, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size)
        x = self.conv_blocks(out)
        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv1d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.25)]
            if bn:
                block.append(nn.BatchNorm1d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(args.num_sensor, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        if args.signal_type == 'FD':
            init_size = 1024 // (2*16)
        else:
            init_size = 1024 // 16

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * init_size, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * init_size, args.num_classes-1), nn.Softmax())

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label


def ACGAN(args, op_num):
    cuda = True if torch.cuda.is_available() else False
    best_acc = 0

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

    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()

    # Initialize generator and discriminator
    generator = Generator(args)
    discriminator = Discriminator(args)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        auxiliary_loss.cuda()

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.gan_lr, betas=(0.5, 0.999))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    fake_num = args.target_train * 3  # generating fake samples

    for epoch in range(args.gan_epoch):
        for i, (x, labels) in enumerate(dataloader):
            batch_size = x.shape[0]

            # Adversarial ground truths
            valid_real = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            valid = Variable(FloatTensor(fake_num, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(fake_num, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_x = Variable(x.type(FloatTensor))
            labels = Variable(labels.type(LongTensor)-1)

            #  ---Train Generator---
            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (fake_num, args.latent_dim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, args.num_classes-1, fake_num)))

            # Generate a batch of images
            gen_x = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator(gen_x)
            g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

            g_loss.backward()
            optimizer_G.step()

            #  ---Train Discriminator---
            optimizer_D.zero_grad()

            # Loss for real x
            real_pred, real_aux = discriminator(real_x)
            d_real_loss = (adversarial_loss(real_pred, valid_real) + auxiliary_loss(real_aux, labels)) / 2

            # Loss for fake x
            fake_pred, fake_aux = discriminator(gen_x.detach())
            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            # Calculate discriminator accuracy
            pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            d_acc = np.mean(np.argmax(pred, axis=1) == gt)

            d_loss.backward()
            optimizer_D.step()

            if epoch % 10 == 0:
                print('[Num-{}, Epoch-{}]  [D loss: {:.4f}, Acc: {:.4f}]  [G loss: {:.4f}]'.format(
                    op_num, epoch, d_loss.item(), 100 * d_acc, g_loss.item()))

            # save the generating data of the best acc
            if epoch >= args.gan_epoch * 2 // 3 and d_acc >= best_acc:
                best_acc = d_acc
                best_epoch = epoch
                data = []
                lab = []
                data_fault = np.concatenate([real_x.data.cpu().numpy(), gen_x.data.cpu().numpy()], axis=0)
                lab_fault = gt
                for i in range(len(lab_fault)):
                    data.append(data_fault[i])
                    lab.append(lab_fault[i]+1)

    print('\nNum-{}, Best_epoch {}, Best_acc {}'.format(op_num, best_epoch, best_acc))

    # creat the saving file
    save_dir = os.path.join('./data/save_dataset/{}_ACGAN'.format(args.transfer_task))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    list_data = [data, lab]
    np.save('{}/{}_{}_{}.npy'.format(save_dir, args.target_train, args.target_train_fault, op_num), list_data)
