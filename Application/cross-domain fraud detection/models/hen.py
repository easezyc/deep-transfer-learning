import torch
import numpy as np

from .layer import FeaturesLinear, MultiLayerPerceptron, FeaturesEmbedding

class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, sequence_lenth, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=2) ** 2
        sum_of_square = torch.sum(x ** 2, dim=2)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=2, keepdim=True)
        return 0.5 * ix


class HENModel(torch.nn.Module):
    """
    A pytorch implementation of Hierarchical Exlainable Network.
    """

    def __init__(self, field_dims, embed_dim, sequence_length, lstm_dims, mlp_dims, dropouts):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.src_embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.tgt_embedding = FeaturesEmbedding(field_dims, embed_dim)

        self.linear = FeaturesLinear(field_dims)
        self.mlp = MultiLayerPerceptron(embed_dim + embed_dim, mlp_dims, dropouts[1])

        self.attention = torch.nn.Embedding(sum(field_dims), 1)
        self.src_attention = torch.nn.Embedding(sum(field_dims), 1)
        self.tgt_attention = torch.nn.Embedding(sum(field_dims), 1)
        # torch.nn.init.constant_(self.attention.weight, 0)
        # torch.nn.init.constant_(self.src_attention.weight, 0)
        # torch.nn.init.constant_(self.tgt_attention.weight, 0)

        self.attr_softmax = torch.nn.Softmax(dim=2)

        self.fm = FactorizationMachine(reduce_sum=False)
        self.src_bn = torch.nn.Sequential(
            torch.nn.BatchNorm1d(sequence_length),
            torch.nn.Dropout(dropouts[0])
        )
        self.tgt_bn = torch.nn.Sequential(
            torch.nn.BatchNorm1d(sequence_length),
            torch.nn.Dropout(dropouts[0])
        )
        self.bn = torch.nn.Sequential(
            torch.nn.BatchNorm1d(sequence_length),
            torch.nn.Dropout(dropouts[0])
        )

        self.event_K = torch.nn.Linear(embed_dim, embed_dim)
        self.event_Q = torch.nn.Linear(embed_dim, embed_dim)
        self.event_V = torch.nn.Linear(embed_dim, embed_dim)

        self.src_event_K = torch.nn.Linear(embed_dim, embed_dim)
        self.src_event_Q = torch.nn.Linear(embed_dim, embed_dim)
        self.src_event_V = torch.nn.Linear(embed_dim, embed_dim)

        self.tgt_event_K = torch.nn.Linear(embed_dim, embed_dim)
        self.tgt_event_Q = torch.nn.Linear(embed_dim, embed_dim)
        self.tgt_event_V = torch.nn.Linear(embed_dim, embed_dim)
        self.event_softmax = torch.nn.Softmax(dim=1)

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
        if dlabel == 'src':
            batch_size = ids.size()[0]

            shared_emb = self.embedding(ids, values)
            src_emb = self.src_embedding(ids, values)

            src_attention = self.attr_softmax(self.src_attention(ids))
            src_event_fea = self.src_bn(torch.mean(src_attention * src_emb, 2) + self.fm(src_emb))

            src_payment_fea = src_event_fea[:, -1, :]
            src_history_fea = src_event_fea[:, :-1, :]

            src_event_K = self.src_event_K(src_history_fea)
            src_event_Q = self.src_event_Q(src_history_fea)
            src_event_V = self.src_event_V(src_history_fea)
            t = torch.sum(src_event_K * src_event_Q, 2, True) / 4 - torch.unsqueeze(seq_mask, 2) * 1e8
            src_his_fea = torch.sum(self.event_softmax(t) * src_event_V, 1)

            shared_attention = self.attr_softmax(self.attention(ids))
            shared_event_fea = self.bn(torch.mean(shared_attention * shared_emb, 2) + self.fm(shared_emb))

            shared_payment_fea = shared_event_fea[:, -1, :]
            shared_history_fea = shared_event_fea[:, :-1, :]

            shared_event_K = self.event_K(shared_history_fea)
            shared_event_Q = self.event_Q(shared_history_fea)
            shared_event_V = self.event_V(shared_history_fea)
            t = torch.sum(shared_event_K * shared_event_Q, 2, True) / 4 - torch.unsqueeze(seq_mask, 2) * 1e8
            shared_his_fea = torch.sum(self.event_softmax(t) * shared_event_V, 1)

            src_term = torch.cat((src_his_fea, src_payment_fea), 1)
            shared_term = torch.cat((shared_his_fea, shared_payment_fea), 1)

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

        elif dlabel == 'tgt':
            batch_size = ids.size()[0]
            shared_emb = self.embedding(ids, values)
            tgt_emb = self.tgt_embedding(ids, values)

            tgt_attention = self.attr_softmax(self.tgt_attention(ids))
            tgt_event_fea = self.tgt_bn(torch.mean(tgt_attention * tgt_emb, 2) + self.fm(tgt_emb))

            tgt_payment_fea = tgt_event_fea[:, -1, :]
            tgt_history_fea = tgt_event_fea[:, :-1, :]

            tgt_event_K = self.tgt_event_K(tgt_history_fea)
            tgt_event_Q = self.tgt_event_Q(tgt_history_fea)
            tgt_event_V = self.tgt_event_V(tgt_history_fea)
            t = torch.sum(tgt_event_K * tgt_event_Q, 2, True) / 4 - torch.unsqueeze(seq_mask, 2) * 1e8
            tgt_his_fea = torch.sum(self.event_softmax(t) * tgt_event_V, 1)

            shared_attention = self.attr_softmax(self.attention(ids))
            shared_event_fea = self.bn(torch.mean(shared_attention * shared_emb, 2) + self.fm(shared_emb))

            shared_payment_fea = shared_event_fea[:, -1, :]
            shared_history_fea = shared_event_fea[:, :-1, :]

            shared_event_K = self.event_K(shared_history_fea)
            shared_event_Q = self.event_Q(shared_history_fea)
            shared_event_V = self.event_V(shared_history_fea)
            t = torch.sum(shared_event_K * shared_event_Q, 2, True) / 4 - torch.unsqueeze(seq_mask, 2) * 1e8
            shared_his_fea = torch.sum(self.event_softmax(t) * shared_event_V, 1)

            tgt_term = torch.cat((tgt_his_fea, tgt_payment_fea), 1)
            shared_term = torch.cat((shared_his_fea, shared_payment_fea), 1)

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
