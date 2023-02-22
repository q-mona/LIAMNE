import numpy as np
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn as nn


import utils
from model import LIAMNE




def get_feature(dataset, device):
    features = np.genfromtxt("data/{}/feature.txt".format(dataset), np.float32)
    return torch.from_numpy(features).to(device)


def get_batches(pairs, neighbors, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size

    for idx in range(n_batches):
        x, y, l, neigh_x, neigh_y = [], [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break

            l.append(pairs[index][0])
            x.append(pairs[index][1])
            y.append(pairs[index][2])
            neigh_x.append(neighbors[pairs[index][1]])
            neigh_y.append(neighbors[pairs[index][2]])
        yield torch.LongTensor(l), torch.LongTensor(x), torch.LongTensor(y),  torch.LongTensor(neigh_x), torch.LongTensor(neigh_y)


def train(args):

    train_data, test_data, valid_data, node_num, layer_num, target_layer, origin_neighs = utils.load_data(
        args.dataset, args.dimensions)
    nodes = [i for i in range(node_num+1)]

    print('layer num: {}, target layer: {}'.format(
        node_num, layer_num, target_layer))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.feature:
        features = get_feature(args.dataset, device)
    else:
        features = None

    m = LIAMNE(node_num+1, layer_num, args.dimensions, device, features)
    m.to(device)


    bce = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(m.parameters(), lr=args.lr)


    for epoch in range(args.epoch):

        if epoch == 0:
            all_neighs = utils.generate_neighs(origin_neighs, args.neighs, layer_num, node_num)
            new_neighs = origin_neighs
        else:
            layer_embs = m.layer_embs.detach().cpu()
            new_network, new_neighs = utils.under_sample(
                nodes, train_data, layer_num, target_layer, layer_embs, args.alpha, args.beta)
            all_neighs = utils.generate_neighs(new_neighs, args.neighs, layer_num, node_num)
        tmp_neighs = torch.LongTensor(all_neighs)

        pairs = []
        for node, neighs in enumerate(all_neighs):
            for layer, neigh in enumerate(neighs):
                temp = [node] + neigh
                pairs.extend(utils.generate_pairs(temp, args.pairs, layer))


        batches = get_batches(pairs, all_neighs, args.batch)
        data_iter = tqdm(
            batches,
            desc="epoch %d" % (epoch),
            total=(len(pairs) + (args.batch - 1)) // args.batch,
            bar_format="{l_bar}{r_bar}",
        )

        for i, pos_pairs in enumerate(data_iter):
            optim.zero_grad()

            final_emb_i = m(
                pos_pairs[0], pos_pairs[1], pos_pairs[3])
            final_emb_j = m(
                pos_pairs[0], pos_pairs[2], pos_pairs[4])
            score = torch.sum(final_emb_i*final_emb_j, dim=1)


            neg_pairs = utils.gen_neg_pairs(
                node_num, layer_num, target_layer, 1, pos_pairs[1], new_neighs)
            neg_pairs = np.array(neg_pairs)

            final_emb_x = m(neg_pairs[:, 0], neg_pairs[:, 2], tmp_neighs[neg_pairs[:, 2]])
            neg_score = torch.sum(final_emb_i*final_emb_x, dim=1)

            labels = torch.ones(len(score)).to(device)
            neg_labels = torch.zeros(len(neg_score)).to(device)

            loss = bce(torch.cat((score, neg_score)),
                       torch.cat((labels, neg_labels)))

            loss.backward()
            optim.step()


        if args.mode == 1:

            layers_t = torch.LongTensor(
                [target_layer for _ in range(node_num+1)]).to(device)
            nodes_t = torch.LongTensor(list(range(node_num+1))).to(device)
            neighs_t = tmp_neighs[nodes_t]

            f_embs = m.forward(layers_t, nodes_t, neighs_t).cpu().detach()

            print('-- link prediction --')
            valid_auc_score = link_prediction(f_embs, valid_data, target_layer, args)
            print('valid auc:', valid_auc_score)
            test_auc_score = link_prediction(f_embs, test_data, target_layer, args)
            print('test auc:', test_auc_score)


        if args.mode == 2:
            print('-- node classification --')
            nodes_t = torch.LongTensor(list(range(node_num+1))).to(device)
            neighs_t = tmp_neighs[nodes_t]
            layers_t = torch.LongTensor([target_layer for _ in range(node_num+1)]).to(device)
            f_embs = m.forward(layers_t, nodes_t, neighs_t).detach().cpu()


            m_embs = torch.zeros_like(f_embs)
            for l in range(layer_num):
                layers_t = torch.LongTensor([l for _ in range(node_num+1)]).to(device)
                m_embs += m.forward(layers_t, nodes_t, neighs_t).detach().cpu()

            macro_f1, micro_f1 = node_classification(args, m_embs/layer_num)

        print('-'*25)


def link_prediction(f_embs, data, target_layer, args):
    test_x, test_y = [], []

    for i, j, w in data[target_layer]:
        test_x.append([i, j])
        test_y.append(w)

    def pdt(x, y):
        return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    auc_score = metrics.roc_auc_score(
        test_y, [pdt(f_embs[i], f_embs[j]) for i, j in test_x])

    return auc_score


def node_classification(args, embs):
    labels = np.genfromtxt('data/{}/label.txt'.format(args.dataset), np.int32)
    labels = torch.from_numpy(labels)
    xent = nn.CrossEntropyLoss()
    train_n, val_n, test_n = [], [], []
    nodes = len(embs)
    for n in range(nodes):
        a = np.random.rand()
        if a < 0.8:
            train_n.append(n)
        elif a < 0.9:
            val_n.append(n)
        else:
            test_n.append(n)

    train_n = torch.LongTensor(train_n)
    val_n = torch.LongTensor(val_n)
    test_n = torch.LongTensor(test_n)
    train_embs = embs[train_n]
    test_embs = embs[test_n]
    val_embs = embs[val_n]

    train_lbls = torch.argmax(labels[train_n], dim=1)
    val_lbls = torch.argmax(labels[val_n], dim=1)
    test_lbls = torch.argmax(labels[test_n], dim=1)

    micro_f1s = []
    macro_f1s = []

    for _ in range(50):
        log = LogReg(embs.shape[1], labels.shape[1])
        opt = torch.optim.Adam(log.parameters(), lr=0.1)
        log.to(train_lbls.device)

        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        for _ in range(50):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_f1_macro = metrics.f1_score(
                val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = metrics.f1_score(
                val_lbls.cpu(), preds.cpu(), average='micro')

            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_f1_macro = metrics.f1_score(
                test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = metrics.f1_score(
                test_lbls.cpu(), preds.cpu(), average='micro')

            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)

        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

    print('marco f1:', np.mean(macro_f1s))
    print('micro f1:', np.mean(micro_f1s))
    return np.mean(macro_f1s), np.mean(micro_f1s)


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


if __name__ == '__main__':
    args = utils.parse_args()
    train(args)
