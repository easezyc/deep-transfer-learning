import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Meta_Linear(torch.nn.Linear): #used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Meta_Linear, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Meta_Linear, self).forward(x)
        return out

class Meta_Embedding(torch.nn.Embedding): #used in MAML to forward input with fast weight
    def __init__(self, num_embedding, embedding_dim):
        super(Meta_Embedding, self).__init__(num_embedding, embedding_dim)
        self.weight.fast = None

    def forward(self, x):
        if self.weight.fast is not None:
            out = F.embedding(
            x, self.weight.fast, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            out = F.embedding(
            x, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out

class Emb(nn.Module):
    def __init__(self, col_names, max_idxs, embedding_size=4, use_cuda=True):
        """
        fnames: feature names
        max_idxs: array of max_idx of each feature
        embedding_size: size of embedding
        dropout: prob for dropout, set None if no dropout
        use_cuda: bool, True for gpu or False for cpu
        """
        super(Emb, self).__init__()
        self.static_emb = StEmb(col_names['static'], max_idxs['static'], embedding_size, use_cuda)
        self.ad_emb = StEmb(col_names['ad'], max_idxs['ad'], embedding_size, use_cuda)
        self.dynamic_emb = DyEmb(col_names['dynamic'], max_idxs['dynamic'], embedding_size, use_cuda)
        self.col_names = col_names
        self.col_length_name = [x + '_length' for x in col_names['dynamic']]

    def forward(self, x):
        static_emb = self.static_emb(x[self.col_names['static']])
        dynamic_emb = self.dynamic_emb(x[self.col_names['dynamic']], x[self.col_length_name])
        concat_embeddings = torch.cat([static_emb, dynamic_emb], 1)
        ad_emb = self.ad_emb(x[self.col_names['ad']])
        #concat_embeddings = static_emb

        return concat_embeddings, ad_emb

class DyEmb(nn.Module):
    def __init__(self, fnames, max_idxs, embedding_size=4, use_cuda=True):
        """
        fnames: feature names
        max_idxs: array of max_idx of each feature
        embedding_size: size of embedding
        dropout: prob for dropout, set None if no dropout
        method: 'avg' or 'sum'
        use_cuda: bool, True for gpu or False for cpu
        """
        super(DyEmb, self).__init__()

        self.fnames = fnames
        self.max_idxs = max_idxs
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda

        # initial layer
        self.embeddings = nn.ModuleList(
            [Meta_Embedding(max_idx + 1, self.embedding_size) for max_idx in self.max_idxs.values()])
        #self.embeddings = nn.ModuleList([Meta_Embedding(max_idx, self.embedding_size) for max_idx in self.max_idxs])


    def forward(self, dynamic_ids, dynamic_lengths):
        """
        input: relative id
        dynamic_ids: Batch_size * Field_size * Max_feature_size
        dynamic_lengths: Batch_size * Field_size
        return: Batch_size * Field_size * Embedding_size
        """
        concat_embeddings = []
        for i, key in enumerate(self.fnames):
            # B*M
            dynamic_ids_tensor = torch.LongTensor(np.array(dynamic_ids[key].values.tolist()))
            dynamic_lengths_tensor = torch.LongTensor(dynamic_lengths[key + '_length'].values.astype(int))
            if self.use_cuda:
                dynamic_ids_tensor = dynamic_ids_tensor.cuda()

            batch_size = dynamic_ids_tensor.size()[0]

            # embedding layer B*M*E
            dynamic_embeddings_tensor = self.embeddings[i](dynamic_ids_tensor)

            # average B*M*E --AVG--> B*E

            dynamic_lengths_tensor = dynamic_lengths_tensor.unsqueeze(1)
            mask = (torch.arange(dynamic_embeddings_tensor.size(1))[None, :] < dynamic_lengths_tensor[:, None]).type(torch.cuda.FloatTensor)
            mask = mask.squeeze(1).unsqueeze(2)
            dynamic_embedding = dynamic_embeddings_tensor.masked_fill(mask == 0, 0)
            dynamic_lengths_tensor[dynamic_lengths_tensor == 0] = 1
            dynamic_embedding = (dynamic_embedding.sum(dim=1) / dynamic_lengths_tensor.cuda()).unsqueeze(1)

            concat_embeddings.append(dynamic_embedding.view(batch_size, 1, self.embedding_size))
        # B*F*E
        concat_embeddings = torch.cat(concat_embeddings, 1)
        return concat_embeddings

class StEmb(nn.Module):
    def __init__(self, col_names, max_idxs, embedding_size=4, use_cuda=True):
        """
        fnames: feature names
        max_idxs: array of max_idx of each feature
        embedding_size: size of embedding
        dropout: prob for dropout, set None if no dropout
        use_cuda: bool, True for gpu or False for cpu
        """
        super(StEmb, self).__init__()
        self.col_names = col_names
        self.max_idxs = max_idxs
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda
        # initial layer
        self.embeddings = nn.ModuleList(
            [Meta_Embedding(max_idx + 1, self.embedding_size) for max_idx in self.max_idxs.values()])

    def forward(self, static_ids):
        """
        input: relative id
        static_ids: Batch_size * Field_size
        return: Batch_size * Field_size * Embedding_size
        """
        concat_embeddings = []
        batch_size = static_ids.shape[0]
        for i, key in enumerate(self.col_names):
            # B*1
            static_ids_tensor = torch.LongTensor(static_ids[key].values.astype(int))
            if self.use_cuda:
                static_ids_tensor = static_ids_tensor.cuda()

            static_embeddings_tensor = self.embeddings[i](static_ids_tensor)

            concat_embeddings.append(static_embeddings_tensor.view(batch_size, 1, self.embedding_size))
        # B*F*E
        concat_embeddings = torch.cat(concat_embeddings, 1)

        return concat_embeddings

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(Meta_Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embed_dim
        if output_layer:
            layers.append(Meta_Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x)

class WideAndDeepModel(torch.nn.Module):
    """
    A pytorch implementation of wide and deep learning.
    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, col_names, max_ids, embed_dim, mlp_dims, dropout, use_cuda, num_expert, num_output):
        super().__init__()
        self.embedding = Emb(col_names, max_ids, embed_dim, use_cuda)
        self.embed_output_dim = (len(col_names['static']) + len(col_names['dynamic'])) * embed_dim
        self.ad_embed_dim = embed_dim * (1 + len(col_names['ad']))
        expert = []
        for i in range(num_expert):
            expert.append(MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, False))
        self.mlp = nn.ModuleList(expert)
        output_layer = []
        for i in range(num_output):
            output_layer.append(Meta_Linear(mlp_dims[-1], 1))
        self.output_layer = nn.ModuleList(output_layer)
        self.attention_layer = torch.nn.Sequential(Meta_Linear(self.ad_embed_dim, mlp_dims[-1]),
                                                   torch.nn.ReLU(),
                                                   Meta_Linear(mlp_dims[-1], num_expert),
                                                   torch.nn.Softmax(dim=1))
        self.output_attention_layer = torch.nn.Sequential(Meta_Linear(self.ad_embed_dim, mlp_dims[-1]),
                                                   torch.nn.ReLU(),
                                                   Meta_Linear(mlp_dims[-1], num_output),
                                                   torch.nn.Softmax(dim=1))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        emb, ad_emb = self.embedding(x)
        ad_emb = torch.cat([torch.mean(emb, 1, True), ad_emb], 1)
        fea = 0
        att = self.attention_layer(ad_emb.view(-1, self.ad_embed_dim))
        for i in range(len(self.mlp)):
            fea += (att[:, i].unsqueeze(1) * self.mlp[i](emb.view(-1, self.embed_output_dim)))
        result = 0

        att = self.output_attention_layer(ad_emb.view(-1, self.ad_embed_dim))
        for i in range(len(self.output_layer)):
            result += (att[:, i].unsqueeze(1) * torch.sigmoid(self.output_layer[i](fea)))
        return result.squeeze(1)