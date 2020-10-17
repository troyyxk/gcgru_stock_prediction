# -*- coding: utf-8 -*-

import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import numpy.linalg as la


def normalized_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(
        d_mat_inv_sqrt).tocoo()
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj


def sparse_to_tuple(mx):
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    L = tf.SparseTensor(coords, mx.data, mx.shape)
    return tf.sparse_reorder(L)


def calculate_laplacian(adj, lambda_max=1):
    #     print("-------------------------")
    #     print(type(adj))
    #     print(adj.shape)
    #     print(adj)
    adj = normalized_adj(adj + sp.eye(adj.shape[0]))
#     print("-------------------------")
#     print(type(adj))
#     print(adj.shape)
#     print(adj)
#     print("-------------------------")
    adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    return sparse_to_tuple(adj)


def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)

    return tf.Variable(initial, name=name)


def evaluation(a, b):
    F_norm = la.norm(a-b, 'fro')/la.norm(a, 'fro')
    return 1-F_norm


def get_trend(pre, cur):
    trends = []
    for i in range(len(pre)):
        if cur[i, 0] - pre[i, 0] > 0:
            trends.append(1)
        else:
            trends.append(0)
    return np.array(trends)


def get_vague_trend(pre, cur, th):
    trends = []
    for i in range(len(pre)):
        if cur[i, 0] - pre[i, 0] > 0:
            if abs(cur[i, 0] - pre[i, 0]) < th:
                trends.append(-2)
            else:
                trends.append(2)
        else:
            if abs(cur[i, 0] - pre[i, 0]) < th:
                trends.append(-1)
            else:
                trends.append(1)
    return np.array(trends)


def avg_relative_error(actual, pred):
    total = 0
    for i in range(len(pred)):
        total += abs(pred[i, 0]-actual[i, 0])/actual[i, 0]
    return total/len(pred)


def get_total_relative_error(actual, pred):
    total = 0
    for i in range(len(pred)):
        total += abs(pred[i, 0]-actual[i, 0])/actual[i, 0]
    return total
