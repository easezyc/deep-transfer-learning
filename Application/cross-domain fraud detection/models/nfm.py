import torch

from .layer import FeaturesLinear, MultiLayerPerceptron, FeaturesEmbedding, FactorizationMachine


class NeuralFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.
    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropouts):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.src_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.tgt_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=False)
        # self.fm1 = FactorizationMachine1(reduce_sum=False)
        self.bn = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.tgt_bn = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.src_bn = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_dims, dropouts[1])  # + hidden_size  + lstm_dims
        self.embed_dim = embed_dim
        self.src_domain_K = torch.nn.Linear(16, 16)
        self.src_domain_Q = torch.nn.Linear(16, 16)
        self.src_domain_V = torch.nn.Linear(16, 16)

        self.tgt_domain_K = torch.nn.Linear(16, 16)
        self.tgt_domain_Q = torch.nn.Linear(16, 16)
        self.tgt_domain_V = torch.nn.Linear(16, 16)

    def forward(self, ids, values, seq_lengths, seq_mask, dlabel):
        """
        :param
        ids: the ids of fields (batch_size, seqlength, fields)
        values: the values of fields (batch_size, seqlength, fields)
        seq_length: the length of historical events (batch_size, 1)
        seq_mask: the attention mask for historical events (batch_size, seqlength)
        dlabel: the domain label of the batch samples (batch_size, 1)
        :return
        torch.sigmoid(result.squeeze(1)): the predition of the target payment
        term: the sequence embedding, output of user behavior extractor (batch_size, 32)
        """
        if dlabel == 'src':
            batch_size = ids.size()[0]
            shared_emb = self.embedding(ids, values)
            shared_term = self.bn(self.fm(shared_emb[:, :, :].view(batch_size, -1, self.embed_dim)))

            src_emb = self.src_embedding(ids, values)
            src_term = self.src_bn(self.fm(src_emb[:, :, :].view(batch_size, -1, self.embed_dim)))

            src_K = self.src_domain_K(src_term)
            src_Q = self.src_domain_Q(src_term)
            src_V = self.src_domain_V(src_term)
            src_a = torch.exp(torch.sum(src_K * src_Q, 1, True) / 4)

            shared_K = self.src_domain_K(shared_term)
            shared_Q = self.src_domain_Q(shared_term)
            shared_V = self.src_domain_V(shared_term)
            shared_a = torch.exp(torch.sum(shared_K * shared_Q, 1, True) / 4)

            term = src_a / (src_a + shared_a) * src_V + shared_a / (src_a + shared_a) * shared_V

            result = self.linear(ids[:, -1, :].view(batch_size, -1)) + self.mlp(term)
            return torch.sigmoid(result.squeeze(1)), term

        elif dlabel == 'tgt':
            batch_size = ids.size()[0]
            shared_emb = self.embedding(ids, values)
            shared_term = self.bn(self.fm(shared_emb[:, :, :].view(batch_size, -1, self.embed_dim)))

            tgt_emb = self.tgt_embedding(ids, values)
            tgt_term = self.tgt_bn(self.fm(tgt_emb[:, :, :].view(batch_size, -1, self.embed_dim)))

            tgt_K = self.tgt_domain_K(tgt_term)
            tgt_Q = self.tgt_domain_Q(tgt_term)
            tgt_V = self.tgt_domain_V(tgt_term)
            tgt_a = torch.exp(torch.sum(tgt_K * tgt_Q, 1, True) / 4)

            shared_K = self.tgt_domain_K(shared_term)
            shared_Q = self.tgt_domain_Q(shared_term)
            shared_V = self.tgt_domain_V(shared_term)
            shared_a = torch.exp(torch.sum(shared_K * shared_Q, 1, True) / 4)

            term = tgt_a / (tgt_a + shared_a) * tgt_V + shared_a / (tgt_a + shared_a) * shared_V

            result = self.linear(ids[:, -1, :].view(batch_size, -1)) + self.mlp(term)
            return torch.sigmoid(result.squeeze(1)), term
