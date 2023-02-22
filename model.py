import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class LIAMNE(nn.Module):
    def __init__(self, node_num, layer_num, emb_size, com_embs, features):
        super(LIAMNE, self).__init__()
        self.emb_size = emb_size
        self.layer_num = layer_num
        self.features = features

        self.layer_embs = nn.Parameter(
            torch.FloatTensor(node_num, layer_num, emb_size))
        self.neighs_embs = nn.Parameter(
            torch.FloatTensor(node_num, layer_num, emb_size)
        )

        if features is not None:
            self.neigh_emb_trans = nn.Parameter(
                torch.FloatTensor(layer_num, features.shape[1], emb_size))

        self.trans_weights = nn.Parameter(
            torch.FloatTensor(emb_size, emb_size)
        )
        self.trans_weights_s1 = nn.Parameter(
            torch.FloatTensor(emb_size, emb_size)
        )

        self.trans_weights_s2 = nn.Parameter(torch.FloatTensor(emb_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.neighs_embs.data.uniform_(-1, 1)
        self.layer_embs.data.uniform_(-1, 1)
        self.trans_weights.data.normal_(std=1.0 / math.sqrt(self.emb_size))
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.emb_size))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.emb_size))
        if self.features is not None:
            self.neigh_emb_trans.data.normal_(
                std=1.0 / math.sqrt(self.emb_size))


    def forward(self, layers, node_i, neighs_i):
        if self.features == None:
            neighs_emb_i = self.neighs_embs[neighs_i]
        else:
            neighs_emb_i = torch.einsum(
                'bijk,akm->bijam', self.features[neighs_i], self.neigh_emb_trans)

        layer_emb_i = self.layer_embs[node_i, layers, :]

        trans_w = self.trans_weights
        trans_w_s1 = self.trans_weights_s1
        trans_w_s2 = self.trans_weights_s2

        neighs_emb_i_tmp = torch.diagonal(
            neighs_emb_i, dim1=1, dim2=3).permute(0, 3, 1, 2)
        neighs_emb_i_tmp = torch.sum(neighs_emb_i_tmp, dim=2)
        attention_i = torch.softmax(
            torch.matmul(
                torch.tanh(torch.matmul(
                    neighs_emb_i_tmp, trans_w_s1)), trans_w_s2
            ).squeeze(2),
            dim=1,
        ).unsqueeze(1)
        neighs_emb_i_tmp = torch.matmul(attention_i, neighs_emb_i_tmp)
        node_embed_i = torch.matmul(neighs_emb_i_tmp, trans_w).squeeze(1)

        final_emb_i = node_embed_i+layer_emb_i

        return F.normalize(final_emb_i, dim=1)
