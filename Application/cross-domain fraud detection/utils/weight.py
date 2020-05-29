import numpy as np
import torch


def convert_to_onehot(label, batch_size, class_num):
    return torch.zeros(batch_size, class_num).cuda().scatter_(1, label, 1)


class Weight:

    @staticmethod
    def cal_weight(s_label, t_label, batch_size=512, class_num=2):
        batch_size = s_label.size()[0]
        s_sca_label = s_label
        s_vec_label = convert_to_onehot(s_sca_label.view(-1, 1), batch_size, class_num)
        s_sum = torch.sum(s_vec_label, dim=0).view(1, class_num)
        s_sum[s_sum == 0] = 10000
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label
        t_vec_label = convert_to_onehot(t_sca_label.view(-1, 1), batch_size, class_num)
        t_sum = torch.sum(t_vec_label, dim=0).view(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        weight_ss = torch.zeros(batch_size, batch_size).cuda()
        weight_tt = torch.zeros(batch_size, batch_size).cuda()
        weight_st = torch.zeros(batch_size, batch_size).cuda()

        set_s = set(s_sca_label.cpu().numpy())
        set_t = set(t_sca_label.cpu().numpy())
        count = 0
        for i in range(class_num):
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i]
                t_tvec = t_vec_label[:, i]
                ss = torch.mm(s_tvec.view(-1, 1), s_tvec.view(1, -1))
                weight_ss = weight_ss + ss
                tt = torch.mm(t_tvec.view(-1, 1), t_tvec.view(1, -1))
                weight_tt = weight_tt + tt
                st = torch.mm(s_tvec.view(-1, 1), t_tvec.view(1, -1))
                weight_st = weight_st + st
                count += 1

        length = count
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = torch.zeros(1)
            weight_tt = torch.zeros(1)
            weight_st = torch.zeros(1)
        return weight_ss, weight_tt, weight_st
