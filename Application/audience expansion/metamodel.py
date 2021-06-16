import torch
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np
from model import WideAndDeepModel

class MetaModel(torch.nn.Module):
    def __init__(self, col_names, max_ids, embed_dim, mlp_dims, dropout, use_cuda, local_lr, global_lr,
                 weight_decay, base_model_name, num_expert, num_output):
        super(MetaModel, self).__init__()
        if base_model_name == 'WD':
            self.model = WideAndDeepModel(col_names = col_names, max_ids = max_ids, embed_dim = embed_dim,
                                     mlp_dims = mlp_dims, dropout = dropout, use_cuda = use_cuda, num_expert=num_expert, num_output = num_output)
        self.local_lr = local_lr
        self.criterion = torch.nn.BCELoss()
        self.meta_optimizer = torch.optim.Adam(params=self.model.parameters(), lr=global_lr, weight_decay=weight_decay)

    def forward(self, x):
        return self.model(x)

    def local_update(self, support_set_x, support_set_y):
        batch_size = support_set_x.shape[0]
        fast_parameters = list(self.model.parameters())
        for weight in fast_parameters:
            weight.fast = None
        support_set_y_pred = self.model(support_set_x)
        label = torch.from_numpy(support_set_y.astype('float32')).cuda()
        loss = self.criterion(support_set_y_pred, label)

        self.model.zero_grad()
        grad = torch.autograd.grad(loss, fast_parameters, create_graph=True, allow_unused=True)
        fast_parameters = []
        for k, weight in enumerate(self.model.parameters()):
            if grad[k] is None:
                continue
            # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
            if weight.fast is None:
                weight.fast = weight - self.local_lr * grad[k]  # create weight.fast
            else:
                weight.fast = weight.fast - self.local_lr * grad[k]
            fast_parameters.append(weight.fast)

        return loss

    def global_update(self, support_set_xs, support_set_ys, query_set_xs, query_set_ys):
        batch_sz = len(support_set_xs)
        losses_q = []
        for i in range(batch_sz):
            loss_sup = self.local_update(support_set_xs[i], support_set_ys[i])
            query_set_y_pred = self.model(query_set_xs[i])
            label = torch.from_numpy(query_set_ys[i].astype('float32')).cuda()

            loss_q = self.criterion(query_set_y_pred, label)
            losses_q.append(loss_q)
        losses_q = torch.stack(losses_q).mean(0)
        self.meta_optimizer.zero_grad()
        losses_q.backward()
        self.meta_optimizer.step()
        fast_parameters = list(self.model.parameters())
        for weight in fast_parameters:
            weight.fast = None
        return losses_q
