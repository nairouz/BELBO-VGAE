#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Authors  : Nairouz Mrabah (mrabah.nairouz@courrier.uqam.ca) & Fawzi Touati (touati.mohamed_fawzi@courrier.uqam.ca)
# @Link    : github.com/nairouz/GMM_VGAE
# @Paper   : Beyond the Evidence Lower Bound: A Contrastive Variatonal Graph Auto-Encoder for Attributed Graph Clustering

# @License : MIT License

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score
f1 = f1_score

def cos_grad(grad1, grad2):
    grad1_list = []
    grad2_list = []
    for i in range(len(grad1)):
        grad1_list.append(grad1[i].flatten())
        grad2_list.append(grad2[i].flatten())
    grad1_vector = torch.cat(grad1_list, 0)
    grad2_vector = torch.cat(grad2_list, 0)
    #grad1_vector = np.concatenate(grad1_list)
    #grad2_vector = np.concatenate(grad2_list)
    #return np.matmul(grad1_vector, grad2_vector) / ((np.linalg.norm(grad1_vector)) * (np.linalg.norm(grad2_vector)))
    return torch.dot(grad1_vector, grad2_vector) / ((torch.norm(grad1_vector)) * (torch.norm(grad2_vector)))

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
       accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size