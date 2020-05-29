import torch
import numpy as np
from .layer import FeaturesLinear, MultiLayerPerceptron, FeaturesEmbedding


class WideAndDeepModel(torch.nn.Module):
    """
    A pytorch implementation of wide and deep learning.
    Reference:
        HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.src_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.tgt_embedding = FeaturesEmbedding(field_dims, embed_dim)

        self.embed_output_dim = 11 * 56 * embed_dim
        self.layer = torch.nn.Linear(self.embed_output_dim, 32)
        self.src_layer = torch.nn.Linear(self.embed_output_dim, 32)
        self.tgt_layer = torch.nn.Linear(self.embed_output_dim, 32)

        self.linear = FeaturesLinear(field_dims)
        self.mlp = MultiLayerPerceptron(32, mlp_dims, dropout)

        self.src_domain_K = torch.nn.Linear(32, 32)
        self.src_domain_Q = torch.nn.Linear(32, 32)
        self.src_domain_V = torch.nn.Linear(32, 32)

        self.tgt_domain_K = torch.nn.Linear(32, 32)
        self.tgt_domain_Q = torch.nn.Linear(32, 32)
        self.tgt_domain_V = torch.nn.Linear(32, 32)

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
        batch_size = ids.size()[0]
        if dlabel == 'src':
            shared_emb = self.embedding(ids, values).view(batch_size, -1)
            shared_term = self.layer(shared_emb)

            src_emb = self.src_embedding(ids, values).view(batch_size, -1)
            src_term = self.src_layer(src_emb)

            src_K = self.src_domain_K(src_term)
            src_Q = self.src_domain_Q(src_term)
            src_V = self.src_domain_V(src_term)
            src_a = torch.exp(torch.sum(src_K * src_Q, 1, True) / 6)

            shared_K = self.src_domain_K(shared_term)
            shared_Q = self.src_domain_Q(shared_term)
            shared_V = self.src_domain_V(shared_term)
            shared_a = torch.exp(torch.sum(shared_K * shared_Q, 1, True) / 6)

            term = src_a / (src_a + shared_a) * src_V + shared_a / (src_a + shared_a) * shared_V

            result = self.linear(ids[:, -1, :].view(batch_size, -1)) + self.mlp(term)
            return torch.sigmoid(result.squeeze(1)), term

        if dlabel == 'tgt':
            shared_emb = self.embedding(ids, values).view(batch_size, -1)
            shared_term = self.layer(shared_emb)

            tgt_emb = self.tgt_embedding(ids, values).view(batch_size, -1)
            tgt_term = self.tgt_layer(tgt_emb)

            tgt_K = self.tgt_domain_K(tgt_term)
            tgt_Q = self.tgt_domain_Q(tgt_term)
            tgt_V = self.tgt_domain_V(tgt_term)
            tgt_a = torch.exp(torch.sum(tgt_K * tgt_Q, 1, True) / 6)

            shared_K = self.tgt_domain_K(shared_term)
            shared_Q = self.tgt_domain_Q(shared_term)
            shared_V = self.tgt_domain_V(shared_term)
            shared_a = torch.exp(torch.sum(shared_K * shared_Q, 1, True) / 6)

            term = tgt_a / (tgt_a + shared_a) * tgt_V + shared_a / (tgt_a + shared_a) * shared_V

            result = self.linear(ids[:, -1, :].view(batch_size, -1)) + self.mlp(term)
            return torch.sigmoid(result.squeeze(1)), term
