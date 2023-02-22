from platform import node
from gensim.models import Word2Vec
import networkx as nx
import numpy as np
from collections import defaultdict
from sklearn import neighbors
import torch
from sqlalchemy import false
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='imdb')

    parser.add_argument('--mode', type=int, default=1, help='1 for link prediction and 2 for node classification')

    parser.add_argument('--feature', type=bool, default=True)

    parser.add_argument('--dimensions', type=int, default=64)

    parser.add_argument('--epoch', type=int, default=30)

    parser.add_argument('--batch', type=int, default=128)

    parser.add_argument('--alpha', type=float, default=0.2)

    parser.add_argument('--beta', type=float, default=0.6)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--neighs', type=int, default=10)

    parser.add_argument('--pairs', type=int, default=3)

    return parser.parse_args()


def load_data(dataset, dimensions):
    train_data = defaultdict(list)
    node_num = 0

    train_edges = np.genfromtxt('data/{}/train.txt'.format(dataset), np.int32)
    for l, i, j in train_edges:
        train_data[l-1].append([i, j, 1])
        node_num = max(node_num, i, j)


    neighs = [defaultdict(list) for _ in range(node_num+1)]
    for l, i, j in train_edges:
        neighs[i][l-1].append(j)
        neighs[j][l-1].append(i)


    test_data = defaultdict(list)
    test_edges = np.genfromtxt('data/{}/test.txt'.format(dataset), np.int32)
    for l, i, j, w in test_edges:
        test_data[l-1].append([i, j, w])
        node_num = max(node_num, i, j)

    valid_data = defaultdict(list)
    if dataset not in ['']:
        valid_edges = np.genfromtxt(
            'data/{}/valid.txt'.format(dataset), np.int32)
        for l, i, j, w in valid_edges:
            valid_data[l-1].append([i, j, w])
            node_num = max(node_num, i, j)

    layer_num = len(train_data)

    target_layer = 0
    target_size = len(train_data[0])
    for layer, edges in train_data.items():
        if target_size > len(edges):
            target_layer = layer
            target_size = len(edges)

    return train_data, test_data, valid_data, node_num, layer_num, target_layer, neighs


def get_G_from_edges(edges):
    edge_dict = defaultdict(set)
    for edge in edges:
        u, v = str(edge[0]), str(edge[1])
        edge_dict[u].add(v)
        edge_dict[v].add(u)
    return edge_dict


def under_sample(nodes, train_data, layer_num, target_layer, embs, alpha, beta):
    target_nodes = set()

    for i, j, w in train_data[target_layer]:
        if w == 1:
            target_nodes.add(i)
            target_nodes.add(j)


    train_pairs = []
    new_network = defaultdict(list)
    new_neighs = [defaultdict(list) for _ in nodes]
    for layer, edges in train_data.items():
        if layer == target_layer:
            for i, j, w in edges:
                train_pairs.append([layer, i, j, w])
                new_neighs[i][layer].append(j)
                new_neighs[j][layer].append(i)
        else:
            for i, j, w in edges:
                if i in target_nodes and j in target_nodes:
                    train_pairs.append([layer, i, j, w])
                    new_neighs[i][layer].append(j)
                    new_neighs[j][layer].append(i)
                else:
                    # r_layer = np.random.randint(0, layer_num)
                    # p = torch.sigmoid(
                    #     torch.sum(embs[i][r_layer]*embs[j][r_layer]))
                    p = torch.sigmoid(
                        torch.sum(embs[i][target_layer]*embs[j][target_layer]))

                    if p > beta:
                        train_pairs.append([layer, i, j, w])
                        new_neighs[i][layer].append(j)
                        new_neighs[j][layer].append(i)
                    elif p >= alpha and np.random.rand() < p:
                        train_pairs.append([layer, i, j, w])
                        new_neighs[i][layer].append(j)
                        new_neighs[j][layer].append(i)

    for l, i, j, w in train_pairs:
        new_network[l].append([i, j, w])
    return new_network, new_neighs


def gen_neg_pairs(node_num, layer_num, target_layer, sample_num, pairs, neighs):
    neg_pairs = []
    for i in pairs:
        cnt = 0
        while cnt < sample_num:
            gen_node = np.random.randint(1, node_num+1)
            while gen_node in neighs[i][target_layer]:
                gen_node = np.random.randint(1, node_num+1)
            neg_pairs.append([target_layer, i, gen_node, 0])
            cnt += 1

    return neg_pairs


def generate_neighs(neighs, sample_num, layer_num, node_num):
    sample_neighs = [[[] for __ in range(layer_num)]
                     for _ in range(node_num+1)]
    for i in range(node_num+1):
        for t in range(layer_num):
            l = len(neighs[i][t])
            if l == 0:
                sample_neighs[i][t] = [i] * sample_num
            elif l < sample_num:
                sample_neighs[i][t] = neighs[i][t]
                sample_neighs[i][t].extend(
                    list(np.random.choice(neighs[i][t], size=sample_num-l)))
            elif l > sample_num:
                sample_neighs[i][t] = list(
                    np.random.choice(neighs[i][t], size=sample_num))
            elif l == sample_num:
                sample_neighs[i][t] = neighs[i][t]

    return sample_neighs


def generate_pairs(walks, window_size, layer):
    pairs = []
    for i in range(len(walks)):
        center_node = walks[i]

        for j in range(max(i - window_size, 0), min(i + window_size + 1, len(walks))):
            if i == j:
                continue
            node = walks[j]
            pairs.append([layer, center_node, node])
    return pairs



