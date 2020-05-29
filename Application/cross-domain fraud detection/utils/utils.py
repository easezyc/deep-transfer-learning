import torch
from utils.weight import Weight


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Stoper():

    def __init__(self, early_step):
        self.max = -1
        self.maxindex = 0
        self.l = []
        self.early_step = early_step

    def add(self, x):
        self.l.append(x)
        if x > self.max:
            self.max = x
            self.maxindex = len(self.l) - 1
            return False
        elif len(self.l) > self.early_step:
            if len(self.l) - 1 - self.maxindex >= self.early_step:
                return True
            else:
                return False
        else:
            return False


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
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def cmmd(source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size()[0]
    weight_ss, weight_tt, weight_st = Weight.cal_weight(s_label, t_label, class_num=2)
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    loss = torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
    if torch.isnan(loss):
        return torch.Tensor([0]).cuda()
    # print(loss)
    return loss


def coral(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 1, keepdim=True) - source
    xc = torch.matmul(torch.transpose(xm, 0, 1), xm)

    # target covariance
    xmt = torch.mean(target, 1, keepdim=True) - target
    xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * 4)
    return loss


def euclidian(source, target):
    d = source.data.shape[1]

    # source covariance
    avg_s = torch.mean(source, 0)
    avg_t = torch.mean(target, 0)
    loss = ((avg_s - avg_t) ** 2).sum()
    return loss


def c_euclidian(source, target, s_label, t_label):
    d = source.data.shape[1]
    src_white = source[s_label == 0]
    src_black = source[s_label == 1]

    tgt_white = target[t_label == 0]
    tgt_black = target[t_label == 1]

    loss = 0
    avg_sw = torch.mean(src_white, 0)
    avg_tw = torch.mean(tgt_white, 0)
    loss += ((avg_sw - avg_tw) ** 2).sum()

    avg_sb = torch.mean(src_black, 0)
    avg_tb = torch.mean(tgt_black, 0)
    loss += ((avg_sb - avg_tb) ** 2).sum()

    return loss


def ced(source, target, s_label, t_label):
    src_white = source[s_label == 0]
    src_black = source[s_label == 1]

    tgt_white = target[t_label == 0]
    tgt_black = target[t_label == 1]
    if (not bool(src_white.numel())) or (not bool(src_black.numel())) or (not bool(tgt_white.numel())) or (
            not bool(tgt_black.numel())):
        avg_s = torch.mean(source, 0)
        avg_t = torch.mean(target, 0)
        loss = ((avg_s - avg_t) ** 2).sum()
        return loss
    loss = 0
    avg_sw = torch.mean(src_white, 0)
    avg_tw = torch.mean(tgt_white, 0)
    loss += ((avg_sw - avg_tw) ** 2).sum()

    avg_sb = torch.mean(src_black, 0)
    avg_tb = torch.mean(tgt_black, 0)
    loss += ((avg_sb - avg_tb) ** 2).sum()

    loss /= (((avg_sw - avg_tb) ** 2).sum() + ((avg_sb - avg_tw) ** 2).sum() + ((avg_sw - avg_sb) ** 2).sum() + (
            (avg_tw - avg_tb) ** 2).sum())

    return loss


def nometric(source, target):
    loss = torch.FloatTensor([0]).cuda()

    return loss
