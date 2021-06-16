import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import tqdm
from tensorflow import keras
from metamodel import MetaModel
from sklearn.metrics import roc_auc_score
import pickle
from utils import Averager, Stoper
import copy
import math
import random

class RunLookalike():
    def __init__(self,
                 config,
                 base_model_name
                 ):
        self.use_cuda = config['use_cuda']
        self.base_model_name = base_model_name
        self.batchsize = config['batchsize']
        self.emb_dim = config['emb_dim']
        self.weight_decay = config['weight_decay']
        self.local_train_lr = config['local_train_lr']
        self.local_test_lr = config['local_test_lr']
        self.global_lr = config['global_lr']
        self.epoch = config['epoch']
        self.root_path = config['root_path']
        self.is_meta = config['is_meta']
        self.task_count = config['task_count']
        self.num_output = config['num_output']
        self.sample_method = config['sample_method']
        self.num_expert = config['num_expert']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.train_stage1_path = self.root_path + 'train_stage1.pkl'
        self.train_stage2_path = self.root_path + 'train_stage2.pkl'
        self.test_hot_stage1_path = self.root_path + 'test_hot_stage1.pkl'
        self.test_hot_stage2_path = self.root_path + 'test_hot_stage2.pkl'
        self.test_cold_stage1_path = self.root_path + 'test_cold_stage1.pkl'
        self.test_cold_stage2_path = self.root_path + 'test_cold_stage2.pkl'


        self.static_context_col = ['carrier', 'consumptionAbility', 'LBS', 'age',
                       'education', 'gender', 'house']
        self.dynamic_context_col = ['interest1', 'interest2', 'interest3', 'kw1', 'kw2', 'topic1', 'topic2']#['interest1', 'interest2', 'interest3', 'kw1', 'kw2', 'topic1', 'topic2']
        self.ad_col = ['advertiserId', 'campaignId', 'creativeSize', 'adCategoryId', 'productId', 'productType']
        self.col_length_name = [x + '_length' for x in self.dynamic_context_col]
        self.ad_max_ids = {
            'advertiserId': 78,
            'campaignId': 137,
            'creativeSize': 14,
            'adCategoryId': 39,
            'productId': 32,
            'productType': 3,
        }
        self.static_max_ids = {
            'carrier': 3,
            'consumptionAbility': 2,
            'LBS': 855,
            'age': 5,
            'education': 7,
            'gender': 2,
            'house': 1,
        }
        self.dynamic_max_ids = {
            'interest1': 124,
            'interest2': 82,
            'interest3': 12,
            'kw1': 263312,
            'kw2': 49780,
            'topic1': 10002,
            'topic2': 9984
        }
        self.columns = {
            'static': self.static_context_col,
            'dynamic': self.dynamic_context_col,
            'ad': self.ad_col
        }
        self.max_ids = {
            'static': self.static_max_ids,
            'dynamic': self.dynamic_max_ids,
            'ad': self.ad_max_ids
        }
        self.label_col = 'label'
        self.train_col = self.static_context_col + self.dynamic_context_col + self.col_length_name + self.ad_col # [self.ID_col] + self.item_col + self.context_col
        self.all_col = [self.label_col, 'aid'] + self.static_context_col + self.dynamic_context_col +  self.col_length_name + self.ad_col

    def read_pkl(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    def read_data(self, path):
        data = self.read_pkl(path)
        return data


    def get_train_data(self):
        print('========Reading data========')
        data_train_stage1 = self.read_data(self.train_stage1_path)[self.all_col]
        print('train stage1 {} '.format(data_train_stage1.shape[0]))

        data_train_stage2 = self.read_data(self.train_stage2_path)[self.all_col]
        print('train stage2 {} '.format(data_train_stage2.shape[0]))

        return data_train_stage1, data_train_stage2

    def get_test_hot_data(self):
        print('========Reading data========')

        data_test_hot_stage1 = self.read_data(self.test_hot_stage1_path)[self.all_col]
        print('test hot stage1 {} '.format(data_test_hot_stage1.shape[0]))

        data_test_hot_stage2 = self.read_data(self.test_hot_stage2_path)[self.all_col]
        print('test hot stage2 {} '.format(data_test_hot_stage2.shape[0]))

        return data_test_hot_stage1, data_test_hot_stage2

    def get_test_cold_data(self):
        print('========Reading data========')

        data_test_cold_stage1 = self.read_data(self.test_cold_stage1_path)[self.all_col]
        print('test cold stage1 {} '.format(data_test_cold_stage1.shape[0]))

        data_test_cold_stage2 = self.read_data(self.test_cold_stage2_path)[self.all_col]
        print('test cold stage2 {} '.format(data_test_cold_stage2.shape[0]))

        return data_test_cold_stage1, data_test_cold_stage2

    def get_model(self):
        if self.base_model_name == 'WD':
            model = MetaModel(col_names = self.columns, max_ids = self.max_ids, embed_dim = self.emb_dim,
                                     mlp_dims = self.mlp_dims, dropout = self.dropout, use_cuda = self.use_cuda,
                              local_lr = self.local_train_lr, global_lr = self.global_lr, weight_decay = self.weight_decay,
                              base_model_name = self.base_model_name, num_expert = self.num_expert, num_output = self.num_output)
        else:
            raise ValueError('Unknown base model: ' + self.base_model_name)

        return model.cuda() if self.use_cuda else model

    def get_criterion(self):
        criterion = torch.nn.BCELoss()
        return criterion

    def eval_auc(self, targets, predicts):
        return roc_auc_score(targets, predicts)

    def get_optimizer(self, model):
        optimizer = torch.optim.Adam(params=model.parameters(), lr=self.local_test_lr, weight_decay=self.weight_decay)
        return optimizer

    def train_stage(self, data_train_stage1, data_train_stage2, model, epoch):
        print('Training Epoch {}:'.format(epoch + 1))
        model.train()
        aid_set = list(set(data_train_stage1.aid))
        avg_loss = Averager()
        data_train = data_train_stage1
        n_samples = data_train.shape[0]
        n_batch = int(np.ceil(n_samples / self.batchsize))
        if self.sample_method == 'normal':
            list_prob = []
            for aid in aid_set:
                list_prob.append(data_train_stage1[data_train_stage1.aid == aid].shape[0])
            list_prob_sum = sum(list_prob)
            for i in range(len(list_prob)):
                list_prob[i] = list_prob[i] / list_prob_sum
        elif self.sample_method == 'sqrt':
            list_prob = []
            for aid in aid_set:
                list_prob.append(math.sqrt(data_train_stage1[data_train_stage1.aid == aid].shape[0]))
            list_prob_sum = sum(list_prob)
            for i in range(len(list_prob)):
                list_prob[i] = list_prob[i] / list_prob_sum
        for i_batch in tqdm.tqdm(range(n_batch)):
            if (self.sample_method == 'normal') or (self.sample_method == 'sqrt'):
                batch_aid_set = np.random.choice(aid_set, size=self.task_count, replace=False, p=list_prob)#random.sample(aid_set, 5)
            elif self.sample_method == 'unit':
                batch_aid_set = random.sample(aid_set, self.task_count)

            list_sup_x, list_sup_y, list_qry_x, list_qry_y = list(), list(), list(), list()
            for aid in batch_aid_set:

                batch_sup = data_train[data_train.aid == aid].sample(self.batchsize)
                batch_qry = data_train[data_train.aid == aid].sample(self.batchsize)

                batch_sup_x = batch_sup[self.train_col]
                batch_sup_y = batch_sup[self.label_col].values
                batch_qry_x = batch_qry[self.train_col]
                batch_qry_y = batch_qry[self.label_col].values

                list_sup_x.append(batch_sup_x)
                list_sup_y.append(batch_sup_y)
                list_qry_x.append(batch_qry_x)
                list_qry_y.append(batch_qry_y)

            loss = model.global_update(list_sup_x, list_sup_y, list_qry_x, list_qry_y)
            avg_loss.add(loss.item())
        print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

    def test_train(self, data_train, model, criterion, optimizer, epoch):
        model.train()
        n_samples = data_train.shape[0]
        data_label = data_train[self.label_col]
        data_train = data_train[self.train_col]
        n_batch = int(np.ceil(n_samples / self.batchsize))
        for i_batch in range(n_batch):
            batch_x = data_train.iloc[i_batch * self.batchsize: (i_batch + 1) * self.batchsize]
            batch_y = data_label.iloc[i_batch * self.batchsize: (i_batch + 1) * self.batchsize].values

            pred = model(batch_x)

            label = torch.from_numpy(batch_y.astype('float32')).cuda()
            loss = criterion(pred, label)

            model.zero_grad()
            loss.backward()
            optimizer.step()

    def test_eval(self, data_test, model):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            n_samples_test = data_test.shape[0]
            n_batch_test = int(np.ceil(n_samples_test / self.batchsize))
            for i_batch in range(n_batch_test):
                batch_x = data_test.iloc[i_batch * self.batchsize: (i_batch + 1) * self.batchsize][self.train_col]
                batch_y = data_test.iloc[i_batch * self.batchsize: (i_batch + 1) * self.batchsize][
                    self.label_col].values
                y = model(batch_x)
                targets.extend(batch_y.tolist())
                predicts.extend(y.tolist())
        return targets, predicts

    def test_stage(self, data_test_stage1, data_test_stage2, model, criterion, stage):
        aid_set = set(data_test_stage1.aid)
        init_model = copy.deepcopy(model.state_dict())
        all_targets, all_predicts = list(), list()
        gauc = 0
        avg_precision = 0
        avg_recall = 0
        for aid in aid_set:
            task_test_stage1 = data_test_stage1[data_test_stage1.aid == aid]
            task_test_stage2 = data_test_stage2[data_test_stage2.aid == aid]
            model.load_state_dict(init_model)
            optimizer = self.get_optimizer(model)
            for i in range(2):
                self.test_train(task_test_stage1.sample(frac=1), model, criterion, optimizer, i)
            targets, predicts = self.test_eval(task_test_stage2, model)
            all_targets.extend(targets)
            all_predicts.extend(predicts)
            test_auc = self.eval_auc(targets, predicts)
            topk = int(len(targets) * 0.05)
            targets = np.array(targets)
            predicts = np.array(predicts)
            at_index = set(np.argwhere(targets == 1).reshape(-1).tolist())
            cdd_index = set(np.argpartition(predicts, -topk)[-topk:].tolist())
            precision = len(at_index & cdd_index) / len(cdd_index)
            recall = len(at_index & cdd_index) / len(at_index)
            print('Aid {}; AUC {}; precision {}; recall {}'.format(aid, test_auc, precision, recall))
            gauc += test_auc
            avg_precision += precision
            avg_recall += recall
        auc = self.eval_auc(all_targets, all_predicts)
        model.load_state_dict(init_model)
        result = 'Stage {}; AUC {}; GAUC {}; Precision {}; Recall {};\n'.format(stage, auc, gauc / len(aid_set), avg_precision / len(aid_set), avg_recall / len(aid_set))
        print(result)
        return result

    def main(self):
        model = self.get_model()
        print(model)
        criterion = self.get_criterion()
        data_train_stage1, data_train_stage2 = self.get_train_data()
        for i_epoch in range(self.epoch):
            self.train_stage(data_train_stage1, data_train_stage2, model, i_epoch)
        del data_train_stage1, data_train_stage2
        torch.save(model.state_dict(),
                   'parameter_{}_{}_{}.pkl'.format(self.task_count, self.emb_dim, self.local_train_lr))
        model.load_state_dict(
            torch.load('parameter_{}_{}_{}.pkl'.format(self.task_count, self.emb_dim, self.local_train_lr)))
        print("=========================test hot aid===========================")
        data_test_hot_stage1, data_test_hot_stage2 = self.get_test_hot_data()
        result_hot = self.test_stage(data_test_hot_stage1, data_test_hot_stage2, model, criterion, 'hot')
        del data_test_hot_stage1, data_test_hot_stage2

        print("=========================test cold aid===========================")
        data_test_cold_stage1, data_test_cold_stage2 = self.get_test_cold_data()
        result_cold = self.test_stage(data_test_cold_stage1, data_test_cold_stage2, model, criterion, 'cold')

        file = open('emb{}_result_file'.format(self.emb_dim), 'a+')
        file.write(
            'task_count: {}; embed_size: {}; local_train_lr: {}; local_test_lr: {}; num_expert: {}; num_output: {}\n'.format(
                self.task_count,
                self.emb_dim,
                self.local_train_lr,
                self.local_test_lr,
                self.num_expert, self.num_output))
        file.write(result_hot)
        file.write(result_cold)
        file.close()