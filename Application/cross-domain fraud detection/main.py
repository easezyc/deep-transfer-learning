import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from sklearn.metrics import roc_auc_score

from models.nfm import NeuralFactorizationMachineModel
from models.hen import HENModel
from models.m3r import SeqM3RModel
from models.wd import WideAndDeepModel
from models.lstm4fd import LSTM4FDModel
from data.dataset import Mydataset
import time
from utils.utils import mmd_rbf_noaccelerate, cmmd, coral, euclidian, c_euclidian, nometric, ced
from utils.utils import Stoper, Averager

import math

max_auc = -1
max_auchead = -1
min_loss = 100000


def get_model(name, field_dims):
    """
    name: the name of the target model
    field_dims: the dimensions of fields
    """
    if name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'hen':
        return HENModel(field_dims, embed_dim=16, sequence_length=11, lstm_dims=20,
                        mlp_dims=(64,),
                        dropouts=(0.2, 0.2))
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(64,), dropout=0.2)
    elif name == 'lstm4fd':
        return LSTM4FDModel(field_dims, embed_dim=16, sequence_length=11, lstm_dims=20, mlp_dims=(64,),
                            dropouts=(0.2, 0.2))
    elif name == 'm3r':
        return SeqM3RModel(field_dims, embed_dim=16, sequence_length=11, lstm_dims=20, mlp_dims=(64,),
                           dropouts=(0.2, 0.2))
    else:
        raise ValueError('unknown model name: ' + name)


def train(model, optimizer, src_loader, tgt_loader, valid_loader, criterion, log_interval=1000, val_interval=50, posp=1,
          nagp=0.5, params_cls=0.5, params_da=0.5, da_type='cmmd', max_fpr=0.01):
    global max_auc
    global max_auchead
    global min_loss
    posp = torch.FloatTensor([posp]).cuda()
    nagp = torch.FloatTensor([nagp]).cuda()
    one = torch.FloatTensor([1]).cuda()

    iter_src = iter(src_loader)
    iter_tgt = iter(tgt_loader)
    num_iter = len(src_loader)
    stoper = Stoper()

    avg_all_loss = Averager()
    avg_src_loss = Averager()
    avg_tgt_loss = Averager()
    avg_da_loss = Averager()
    start_time = time.time()
    for i in range(1, num_iter * 20):
        model.train()
        src_ids, src_values, src_seqlength, src_label, src_seq_mask = iter_src.next()
        src_ids, src_values, src_label = src_ids.cuda(), src_values.cuda(), src_label.cuda().float()
        src_seq_mask = src_seq_mask.cuda()
        if i % len(src_loader) == 0:
            iter_src = iter(src_loader)
        if i % len(tgt_loader) == 0:
            iter_tgt = iter(tgt_loader)

        src_p = posp * src_label + nagp * (one - src_label)
        src_y, src_fea_LSTM = model(src_ids, src_values, src_seqlength, src_seq_mask, 'src')
        src_loss = torch.mean(
            src_p * criterion(src_y, src_label))  # + torch.mean(src_p * criterion(src_spey, src_label))

        tgt_ids, tgt_values, tgt_seqlength, tgt_label, tgt_seq_mask = iter_tgt.next()
        tgt_ids, tgt_values, tgt_label = tgt_ids.cuda(), tgt_values.cuda(), tgt_label.cuda().float()
        tgt_seq_mask = tgt_seq_mask.cuda()
        # print(tgt_seqlength, tgt_label)

        tgt_p = posp * tgt_label + nagp * (one - tgt_label)
        tgt_y, tgt_fea_LSTM, tgt_spey = model(tgt_ids, tgt_values, tgt_seqlength, tgt_seq_mask, 'tgt')
        tgt_loss = torch.mean(
            tgt_p * criterion(tgt_y, tgt_label))  # + 0.5 * torch.mean(tgt_p * criterion(tgt_spey, tgt_label))
        if da_type == 'cmmd':
            da_loss = cmmd(src_fea_LSTM, tgt_fea_LSTM, src_label.long(), tgt_label.long())
        elif da_type == 'mmd':
            da_loss = mmd_rbf_noaccelerate(src_fea_LSTM, tgt_fea_LSTM)
        elif da_type == 'coral':
            da_loss = coral(src_fea_LSTM, tgt_fea_LSTM)
        elif da_type == 'euclidian':
            da_loss = euclidian(src_fea_LSTM, tgt_fea_LSTM)
        elif da_type == 'c_euclidian':
            da_loss = c_euclidian(src_fea_LSTM, tgt_fea_LSTM, src_label.long(), tgt_label.long())
        elif da_type == 'nometric':
            da_loss = nometric(src_fea_LSTM, tgt_fea_LSTM)
        elif da_type == 'ced':
            da_loss = ced(src_fea_LSTM, tgt_fea_LSTM, src_label.long(), tgt_label.long())
        lambd = 2 / (1 + math.exp((- 5 * i) / (len(src_loader)))) - 1
        loss = params_cls * src_loss + tgt_loss + params_da * lambd * da_loss
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        avg_all_loss.add(loss.item())
        avg_src_loss.add(src_loss.item())
        avg_tgt_loss.add(tgt_loss.item())
        avg_da_loss.add(da_loss.item())
        if (i + 1) % log_interval == 0:
            print(
                'step: {}, loss: {:.4f}, src_loss: {:.4f}, tgt_loss: {:.4f}, {}_loss:, {:.4f}, lambda: {}'.format(i + 1,
                                                                                                                  avg_all_loss.item(),
                                                                                                                  avg_src_loss.item(),
                                                                                                                  avg_tgt_loss.item(),
                                                                                                                  da_type,
                                                                                                                  avg_da_loss.item(),
                                                                                                                  lambd))
            avg_all_loss = Averager()
            avg_src_loss = Averager()
            avg_tgt_loss = Averager()
            avg_da_loss = Averager()

        if (i + 1) % val_interval == 0:
            end_time = time.time()
            print('train time (s):', end_time - start_time)
            start_time = time.time()
            auc_head, loss, auc = test(model, valid_loader, criterion, posp, max_fpr)
            if loss < min_loss:
                min_loss = loss
            if auc > max_auc:
                max_auc = auc
            if auc_head > max_auchead:
                torch.save(model, f'{save_dir}/tmp.pt')
                max_auchead = auc_head
            print(
                'dev ---  auchead: {:.4f}, max_auchead: {:.4f}, auc: {:.4f}, max_auc: {:.4f}, loss: {:.4f}, minloss: {:.4f}'.format(
                    auc_head, max_auchead, auc, max_auc, loss, min_loss))
            end_time = time.time()
            print('dev time (s):', end_time - start_time)
            start_time = time.time()
            if stoper.add(auc_head):
                print('training end')
                break


def test(model, data_loader, criterion, posp, max_fpr):
    model.eval()
    targets, predicts = list(), list()
    loss = Averager()
    posp = torch.FloatTensor([posp]).cuda()
    one = torch.FloatTensor([1]).cuda()
    with torch.no_grad():
        for j, (ids, values, seqlength, label, seq_mask) in enumerate(data_loader):
            ids, values = ids.cuda(), values.cuda()
            label = label.cuda().float()
            seq_mask = seq_mask.cuda()
            y, _ = model(ids, values, seqlength, seq_mask, 'tgt')
            p = posp * label + (one - posp) * (one - label)
            loss.add(torch.mean(p * criterion(y, label)).item())
            targets.extend(label.tolist())
            predicts.extend(y.tolist())
    model.train()
    return roc_auc_score(targets, predicts, max_fpr=max_fpr), loss.item(), roc_auc_score(targets, predicts)


def main(batch_size,
         log_step,
         val_step,
         posp,
         nagp,
         sample_radio,
         params_cls,
         params_da,
         src_name,
         tgt_name,
         da_type,
         model,
         optimizer,
         max_fpr,
         data_radio):
    # feature = ['item_id', 'user_id']
    global max_auc
    global max_auchead
    global min_loss

    src_train_path = 'd:/Jupyter/zhuyc/data/fourcountry/%s%s/train.txt' % (src_name, data_radio)
    tgt_train_path = 'd:/Jupyter/zhuyc/data/fourcountry/%s%s/train.txt' % (tgt_name, data_radio)
    dev_path = 'd:/Jupyter/zhuyc/data/fourcountry/%s%s/test.txt' % (tgt_name, data_radio)
    test_path = 'd:/Jupyter/zhuyc/data/fourcountry/%s%s/test.txt' % (tgt_name, data_radio)

    src_train_dataset = Mydataset(src_train_path, sample_radio)
    tgt_train_dataset = Mydataset(tgt_train_path, sample_radio)
    dev_dataset = Mydataset(dev_path, 1)
    test_dataset = Mydataset(test_path, 1)
    src_sampler = WeightedRandomSampler(src_train_dataset.get_weight(), len(src_train_dataset), True)
    tgt_sampler = WeightedRandomSampler(tgt_train_dataset.get_weight(), len(tgt_train_dataset), True)
    print('src_name:', src_name, 'tgt_name:', tgt_name, 'src training set:', len(src_train_dataset),
          'tgt training set:', len(tgt_train_dataset), 'dev set:', len(dev_dataset), 'test set:', len(test_dataset))

    src_train_loader = DataLoader(src_train_dataset, batch_size=batch_size, num_workers=1, sampler=src_sampler,
                                  drop_last=True)
    tgt_train_loader = DataLoader(tgt_train_dataset, batch_size=batch_size, num_workers=1, sampler=tgt_sampler,
                                  drop_last=True)

    valid_loader = DataLoader(dev_dataset, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)
    if data_radio != '0.05' and data_radio != '0.1':
        val_step = int(len(tgt_train_dataset) / 1024)
    criterion = torch.nn.BCELoss(reduction='none')
    train(model, optimizer, src_train_loader, tgt_train_loader, valid_loader, criterion, log_step, val_step, posp, nagp,
          params_cls, params_da, da_type, max_fpr)

    model = torch.load(f'{save_dir}/tmp.pt').cuda()
    start_time = time.time()
    auc_head, loss, auc = test(model, test_loader, criterion, posp, max_fpr)
    end_time = time.time()
    print('test time (s):', end_time - start_time)
    print('dev auchead:', max_auchead, 'dev auc:', max_auc, 'dev loss:', min_loss)
    print('test auchead:', auc_head, 'test auc:', auc, 'test loss:', loss)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='hen')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--posp', type=float, default=1)
    parser.add_argument('--nagp', type=float, default=0.5)
    parser.add_argument('--params_cls', type=float, default=1)
    parser.add_argument('--params_da', type=float, default=1)
    parser.add_argument('--sample_radio', type=float, default=5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--data_radio', default='0.05')
    parser.add_argument('--val_step', type=int, default=30)
    parser.add_argument('--weight_decay', type=float, default=1e-8)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--sequence_length', type=int, default=11)
    parser.add_argument('--lstm_dims', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--mlp_dims', default=(64,))
    parser.add_argument('--src_name', default='TH')
    parser.add_argument('--tgt_name', default='VN')
    parser.add_argument('--da_type', default='nometric')
    parser.add_argument('--max_fpr', type=float, default=0.01)
    parser.add_argument('--gpu', type=str, default='3')
    # 判断结果
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    save_dir = 'seqnfmchkpt_gpu%s_%s' % (args.gpu, args.da_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(
        'positive probability: {}, nagative probability: {}, sample radio: {}, batch size: {}, params_cls: {}, params_da: {}, da_type: {}, weight_decay: {}, src_name: {}, tgt_name: {}, gpu: {}'.format(
            args.posp, args.nagp, args.sample_radio, args.batch_size, args.params_cls, args.params_da, args.da_type,
            args.weight_decay, args.src_name, args.tgt_name, args.gpu))

    field_dims = [int(1e5)]
    model = get_model(args.model_name,
                      field_dims).cuda()
    print(model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # torch.manual_seed(args.seed)#为CPU设置随机种子
    # torch.cuda.manual_seed(args.seed)#为当前GPU设置随机种子
    main(args.batch_size,
         args.log_step,
         args.val_step,
         args.posp,
         args.nagp,
         args.sample_radio,
         args.params_cls,
         args.params_da,
         args.src_name,
         args.tgt_name,
         args.da_type,
         model,
         optimizer,
         args.max_fpr,
         args.data_radio)
