
import torch
import math
import torch.nn as nn
import numpy as np


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def convert_to_onehot(sca_label, class_num=31):
    return np.eye(class_num)[sca_label]


def cal_weight(s_label, t_label, class_num=4):
    batch_size = s_label.size()[0]
    s_sca_label = s_label.cpu().data.numpy()
    s_vec_label = convert_to_onehot(s_sca_label, class_num=class_num)
    s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
    s_sum[s_sum == 0] = 100
    s_vec_label = s_vec_label / s_sum

    # t_sca_label = t_label.cpu().data.max(1)[1].numpy()
    t_sca_label = t_label.cpu().data.numpy()
    t_vec_label = convert_to_onehot(t_sca_label, class_num=class_num)
    t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
    t_sum[t_sum == 0] = 100
    t_vec_label = t_vec_label / t_sum

    index = list(set(s_sca_label) & set(t_sca_label))
    mask_arr = np.zeros((batch_size, class_num))
    mask_arr[:, index] = 1
    t_vec_label = t_vec_label * mask_arr
    s_vec_label = s_vec_label * mask_arr

    weight_ss = np.matmul(s_vec_label, s_vec_label.T)
    weight_tt = np.matmul(t_vec_label, t_vec_label.T)
    weight_st = np.matmul(s_vec_label, t_vec_label.T)

    length = len(index)
    if length != 0:
        weight_ss = weight_ss / length
        weight_tt = weight_tt / length
        weight_st = weight_st / length
    else:
        weight_ss = np.array([0])
        weight_tt = np.array([0])
        weight_st = np.array([0])
    return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


def SAM(source, target, s_label, t_label,
        class_num=4, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    batch_size = source.size()[0]
    weight_ss, weight_tt, weight_st = cal_weight(
        s_label, t_label, class_num=class_num)
    weight_ss = torch.from_numpy(weight_ss).cuda()
    weight_tt = torch.from_numpy(weight_tt).cuda()
    weight_st = torch.from_numpy(weight_st).cuda()

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]
    loss = torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)

    # loss_fd = 0
    # for i in range(class_num-1):
    #     source_fault = target[s_label == i+1, :]
    #     source_square = torch.matmul(source_fault, source_fault.T)
    #     distance = torch.sqrt(torch.mean(source_square))
    #     loss_fd += distance
    # mse = torch.nn.MSELoss(reduction='mean')
    # source_fault_center = source_fault.mean(dim=0)
    # source_fault_center = source_fault_center.unsqueeze(0).expand(int(source_fault.size(0)), int(source_fault.size(1)))
    # loss1 = mse(source_fault_center, source_fault)
    # l2_loss = sum(torch.mean(torch.abs(p)) for p in model.parameters())
    # loss = loss_nc + loss_fd

    return loss
