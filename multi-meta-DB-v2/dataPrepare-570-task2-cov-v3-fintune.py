# -*- coding: utf-8 -*-
from numpy.random import seed
import csv
import sqlite3
import time
import numpy as np
import random
import pandas as pd
from pandas import DataFrame
import scipy.sparse as sp
import math
import copy
import pickle

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import KernelPCA

import sys
import torch
# torch.cuda.empty_cache()
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from copy import deepcopy
from myModel import *
from myModel_GAT import *

class focal_loss(nn.Module):
    def __init__(self, gamma=2):
        super(focal_loss,self).__init__()
        self.gamma = gamma
    def forward(self, preds, labels):
        labels = labels.view(-1, 1) # [B * S, 1]
        preds = preds.view(-1, preds.size(-1)) # [B * S, C]
        preds_logsoft = F.log_softmax(preds, dim=1) 
        preds_softmax = torch.exp(preds_logsoft)
        preds_softmax = preds_softmax.gather(1, labels)  
        preds_logsoft = preds_logsoft.gather(1, labels)
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  
        loss = loss.mean()
        return loss

random.seed(0)
device = torch.device('cuda')
# loss_func = focal_loss()
loss_func = F.cross_entropy

def save_result(feature_name, result_type, clf_type, result):
    with open(feature_name + '_' + result_type + '_' + clf_type+ '.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0
def self_metric_calculate(y_true, pred_type):
    y_true = y_true.ravel()
    y_pred = pred_type.ravel()
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_pred_c = y_pred.take([0], axis=1).ravel()
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i in range(len(y_true_c)):
        if (y_true_c[i] == 1) and (y_pred_c[i] == 1):
            TP += 1
        if (y_true_c[i] == 1) and (y_pred_c[i] == 0):
            FN += 1
        if (y_true_c[i] == 0) and (y_pred_c[i] == 1):
            FP += 1
        if (y_true_c[i] == 0) and (y_pred_c[i] == 0):
            TN += 1
    print("TP=", TP, "FN=", FN, "FP=", FP, "TN=", TN)
    return (TP / (TP + FP), TP / (TP + FN))


def multiclass_precision_recall_curve(y_true, y_score):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_score_c = y_score.take([0], axis=1).ravel()
    precision, recall, pr_thresholds = precision_recall_curve(y_true_c, y_score_c)
    return (precision, recall, pr_thresholds)


def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


def drawing(d_result, contrast_list, info_list):
    column = []  
    for i in contrast_list:
        column.append(i)
    df = pd.DataFrame(columns=column)
    if info_list[-1] == 'aupr':
        for i in contrast_list:
            df[i] = d_result[i][:, 1]
    else:
        for i in contrast_list:
            df[i] = d_result[i][:, 2]
    df = df.astype('float')
    color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
    df.plot.box(ylim=[0, 1.0], grid=True, color=color)
    return 0

# def evaluate(pred_type, pred_score, y_test, event_num, set_name):
#     all_eval_type = 11
#     result_all = np.zeros((all_eval_type, 1), dtype=float)
#     each_eval_type = 6
#     result_eve = np.zeros((event_num, each_eval_type), dtype=float)
#     y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
#     pred_one_hot = label_binarize(pred_type, classes=np.arange(event_num))

#     precision, recall, th = multiclass_precision_recall_curve(y_one_hot, pred_score)

#     result_all[0] = accuracy_score(y_test, pred_type)
#     result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
#     result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
# #     result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
# #     result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
#     result_all[5] = f1_score(y_test, pred_type, average='micro')
#     result_all[6] = f1_score(y_test, pred_type, average='macro')
#     result_all[7] = precision_score(y_test, pred_type, average='micro')
#     result_all[8] = precision_score(y_test, pred_type, average='macro')
#     result_all[9] = recall_score(y_test, pred_type, average='micro')
#     result_all[10] = recall_score(y_test, pred_type, average='macro')
#     for i in range(event_num):
#         result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
# #         result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
# #                                           average=None)
# #         result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
# #                                          average=None)
#         result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
#                                     average='binary')
#         result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
#                                            average='binary')
#         result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
#                                         average='binary')
#     return [result_all, result_eve]

def evaluate(pred_type, pred_score, y_test, event_num, set_name):
    all_eval_type = 6
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, classes=np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[3] = f1_score(y_test, pred_type, average='macro')
    result_all[4] = precision_score(y_test, pred_type, average='macro')
    result_all[5] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        # result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),average=None)
        result_eve[i, 2] = 0.0
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]

def print_file(str_, save_file_path=None):
    print(str_)
    if save_file_path != None:
        f = open(save_file_path, 'a')
        print(str_, file=f)
def rename_ddi(df_drug, mechanism,action,drugA,drugB):  #药物、event重命名，生成ddi文件
    name_dict = {}
    label_dict = {}
    count={}
    d_event=[]
    new_label = [] 
    for i in range(len(mechanism)):
        d_event.append(mechanism[i]+" "+action[i])
    for i in d_event:
        if i in count:
            count[i]+=1
        else:
            count[i]=1
    event_temp = sorted(count.items(), key=lambda x: x[1],reverse=True)
    for i in range(len(event_temp)):
        label_dict[event_temp[i][0]]= i
    for i in range(len(d_event)):
        new_label.append(label_dict[d_event[i]])
    for i in range(len(df_drug["name"])):
        name_dict[df_drug["name"][i]] = i
    name_drugA_new = []
    name_drugB_new = []
    name_dfdrug_new = []
    for i in range(len(drugA)):
        name_drugA_new.append(name_dict[drugA[i]])
        name_drugB_new.append(name_dict[drugB[i]])
    for j in range(len(df_drug["name"])):
        name_dfdrug_new.append(name_dict[df_drug["name"][j]])
    df_drug["name"] = name_dfdrug_new
    df_drug.to_csv("df_drug570.csv")
    pd_DDI = pd.DataFrame({"d1":name_drugA_new ,"type":new_label, "d2":name_drugB_new})
    pd_DDI.to_csv("ddis.csv")
    return df_drug, pd_DDI


class Data_triplets():    #
    def __init__(self, idx):
        pd_ddi = pd.read_csv("./ddis.csv", usecols=[1, 2,3], names=None)
        self.ddi_rel_triplets_list = pd_ddi.values.tolist()  #(drugA, type, drugB)
        self.all_drug_dict_pool = {}
        for tri in self.ddi_rel_triplets_list:
            for tid in [0,2]:
                i = tri[tid]
                if i in self.all_drug_dict_pool.keys():
                    self.all_drug_dict_pool[i].append(tri)
                else:
                    self.all_drug_dict_pool[i] = [tri]    

        self.drug_id = list(self.all_drug_dict_pool.keys())   
        random.shuffle(self.drug_id)        
        seen = np.load("570seen_rename.npy", allow_pickle = True)
        unseen = np.load("570unseen_rename.npy", allow_pickle = True)
        self.seen_drugs = seen[idx]
        self.unseen_drugs = unseen[idx]
        
        self.train_set_triplets = []
        self.test_set_triplets = []
        self.others_triplets = []
        for l2 in self.ddi_rel_triplets_list:
            if l2[0] in self.seen_drugs and l2[2] in self.seen_drugs:
                self.train_set_triplets.append(l2)
            elif (l2[0] in self.seen_drugs and l2[2] in self.unseen_drugs) or (l2[0] in self.unseen_drugs and l2[2] in self.seen_drugs):
                self.test_set_triplets.append(l2)
            else:
                self.others_triplets.append(l2)


    def get_drug_related_triplets_dict(self):
        self.train_drug_dict_pool = {}
        self.test_drug_dict_pool = {}

        for tri in self.train_set_triplets:
            for tid in [0,2]:
                i = tri[tid]
                if i in self.train_drug_dict_pool.keys():
                    self.train_drug_dict_pool[i].append(tri)
                else:
                    self.train_drug_dict_pool[i] = [tri]
        for tri in self.test_set_triplets:
            if tri[0] in self.train_drug_dict_pool.keys():
                i = tri[0]
            elif tri[2] in self.train_drug_dict_pool.keys() :
                i = tri[2]
            else:
                continue
            if i in self.test_drug_dict_pool.keys():
                self.test_drug_dict_pool[i].append(tri)
            else:
                self.test_drug_dict_pool[i] = [tri] 


def generate_task_data_train(drug,train_dict, test_dict,train_triple, test_triple,fea_mtx, training = True):  ##新加了传特征矩阵的参数
    if training:
        mol_batch_dict = {}
        support_rate = 0.8
        meta_feat_mtx = []

        triplets = train_dict[drug]

        triplets = np.array(triplets)
        source, r, dest = triplets.transpose()
        uniq_v,edges = np.unique((source, dest), return_inverse=True)
        for i in list(uniq_v):
            meta_feat_mtx.append(fea_mtx[i])

        src, dst = np.reshape(edges, (2, -1))
        random_list = [i for i in range(len(triplets))]
        random.shuffle(random_list)
        src_random = src[random_list]
        dst_random = dst[random_list]
        r_random = r[random_list]

        sup_src = src_random[:int(len(src_random) * support_rate)]
        sup_dst = dst_random[:int(len(src_random) * support_rate)]
        sup_r = r_random[:int(len(src_random) * support_rate)]

        sup_edge_src =  torch.from_numpy(np.concatenate((sup_src, sup_dst)))
        sup_edge_dst = torch.from_numpy(np.concatenate((sup_dst, sup_src)))
        sup_edge = torch.stack((sup_edge_src,sup_edge_dst))
        sup_rel = torch.from_numpy(np.concatenate((sup_r, sup_r)))

        query_src = src_random[int(len(src_random)* support_rate):]
        query_dst = dst_random[int(len(src_random) * support_rate):]
        query_r = r_random[int(len(src_random) * support_rate):]
        
        query_edge_src = torch.from_numpy(np.concatenate((query_src, query_dst)))
        query_edge_dst = torch.from_numpy(np.concatenate((query_dst, query_src)))
        query_edge = torch.stack((query_edge_src, query_edge_dst))
        query_rel = torch.from_numpy(np.concatenate((query_r, query_r)))
        
    else:
        mol_batch_dict = {}
        meta_feat_mtx = []
        triplets_train = train_dict[drug]
        triplets_test = test_dict[drug]
        len_train = len(triplets_train)
#         print("len_train:", len_train)
        triplets = deepcopy(triplets_train)
        for tri in triplets_test:
            triplets.append(tri)
        source, r, dest = np.array(triplets).transpose()
        uniq_v,edges = np.unique((source, dest), return_inverse=True)
        for i in list(uniq_v):
            meta_feat_mtx.append(fea_mtx[i])
        
        src, dst = np.reshape(edges, (2, -1))
        sup_src = src[:len_train]
        sup_dst = dst[:len_train]
        sup_r = r[:len_train]
        sup_edge_src =  torch.from_numpy(np.concatenate((sup_src, sup_dst)))
        sup_edge_dst = torch.from_numpy(np.concatenate((sup_dst, sup_src)))
        sup_edge = torch.stack((sup_edge_src,sup_edge_dst))
        sup_rel = torch.from_numpy(np.concatenate((sup_r, sup_r)))
        
        query_src = src[len_train:]
        query_dst = dst[len_train:]
        query_r = r[len_train:]
        query_edge_src = torch.from_numpy(np.concatenate((query_src, query_dst)))
        query_edge_dst = torch.from_numpy(np.concatenate((query_dst, query_src)))
        query_edge = torch.stack((query_edge_src, query_edge_dst))
        query_rel = torch.from_numpy(np.concatenate((query_r, query_r)))
    return torch.Tensor(meta_feat_mtx),sup_edge,sup_rel,query_edge,query_rel


def data_iter(batch_size, drug_poolList):
    num_drugs = len(drug_poolList)
    indices = list(range(num_drugs))
    random.shuffle(indices)
    for i in range(0, num_drugs, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_drugs)])
        yield drug_poolList.index_select(0, j).tolist()

class Meta(nn.Module):
    def __init__(self, num_inputs_in = 570):
        super(Meta, self).__init__()        
        num_inputs = num_inputs_in
        num_outputs = 47
        num_hiddens = 768
        num_edg_hiddens = 2 *num_hiddens
#         self.update_lr= 0.01 #0.01
        self.update_lr= 0.01 #0.01
        self.meta_lr = 0.001
        self.update_step = 2
        self.update_step_test = 2
        self.net = Mymodel_GAT_cov_gelu(num_inputs, num_outputs, num_hiddens, num_edg_hiddens)
        # self.net = Mymodel_GAT_cov_residue_gelu(num_inputs, num_outputs, num_hiddens, num_edg_hiddens)
#         self.net = Mymodel_GAT_shrink(num_inputs, num_outputs, num_hiddens, num_edg_hiddens)
        # self.net = Mymodel_GAT_drop(num_inputs, num_outputs, num_hiddens, num_edg_hiddens)
        # self.net = Mymodel_GAT_residue(num_inputs, num_outputs, num_hiddens, num_edg_hiddens)
        # self.net =  Mymodel_GAT_noresidue(num_inputs, num_outputs, num_hiddens, num_edg_hiddens)
        # self.net = Mymodel_GAT_gelu(num_inputs, num_outputs, num_hiddens, num_edg_hiddens)
        # self.net = Mymodel(num_inputs, num_outputs, num_hiddens, num_edg_hiddens)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr) # no decay 
#         self.meta_optim = optim.RAdam(self.net.parameters(), lr=self.meta_lr) # no decay 
        # self.meta_optim = optim.SGD(self.net.parameters(), lr=self.meta_lr) # no decay 
        # self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr, weight_decay=0.0001) # no decay 
    def forward(self, drug_pool_batch_list_train,feat_mtrx_sel):
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        qry_sz = [0 for _ in range(self.update_step + 1)]
        for drug in drug_pool_batch_list_train:
            feat_metrix_meta, spt_edg, spt_r, qry_edg, qry_r = generate_task_data_train(drug,train_drug_dict_pool,test_drug_dict_pool,train_set_triplets,test_set_triplets, feat_mtrx_sel)
            feat_metrix_meta, spt_edg, spt_r, qry_edg, qry_r =  feat_metrix_meta.to(device),spt_edg.to(device), spt_r.to(device), qry_edg.to(device), qry_r.to(device)
            logits = self.net(feat_metrix_meta, spt_edg,  vars=None, bn_training=True)
            loss = loss_func (logits, spt_r)
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            for k in range(1, self.update_step):
                logits = self.net(feat_metrix_meta, spt_edg, vars=fast_weights, bn_training=True)
                loss = loss_func (logits,spt_r)
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q = self.net(feat_metrix_meta, qry_edg, fast_weights, bn_training=True)
                loss_q = loss_func (logits_q, qry_r)
                losses_q[k + 1] += loss_q
                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)  #返回的是坐标
                    correct = torch.eq(pred_q, qry_r).sum().cpu().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct
                    qry_sz[k + 1] = qry_sz[k + 1] + len(qry_r)
        loss_q = losses_q[-1] / len(drug_pool_batch_list_train)
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        accs = np.array(corrects[-1]) /qry_sz[-1]
        return accs
    
    def finetune(self,drug_pool_batch_list_test,feat_mtrx_sel):
        corrects = [0 for _ in range(self.update_step_test + 1)]
        qry_sz = [0 for _ in range(self.update_step_test + 1)]
        pred_score_list = [] 
        y_truth_list = [] 
        net = deepcopy(self.net)
        loss_mean = 0.0
        for drug in drug_pool_batch_list_test:
            feat_metrix_meta, spt_edg, spt_r, qry_edg, qry_r = generate_task_data_train(drug,train_drug_dict_pool,test_drug_dict_pool,train_set_triplets,test_set_triplets,feat_mtrx_sel,training = False)
            feat_metrix_meta, spt_edg, spt_r, qry_edg, qry_r =  feat_metrix_meta.to(device),spt_edg.to(device), spt_r.to(device), qry_edg.to(device), qry_r.to(device)
            logits = net(feat_metrix_meta, spt_edg)
            # logits = net(feat_metrix_meta, spt_edg, vars=None, bn_training=False)
            # logits = net(feat_metrix_meta,spt_edg, dropout_rate=0) #drop
            loss = loss_func (logits, spt_r)
            grad = torch.autograd.grad(loss, net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
            for k in range(1, self.update_step_test):
                logits = net(feat_metrix_meta,spt_edg, fast_weights, bn_training=True)
                # logits = net(feat_metrix_meta,spt_edg, fast_weights, bn_training=False)
                # logits = net(feat_metrix_meta,spt_edg, fast_weights, bn_training=True, dropout_rate=0)#drop
                loss = loss_func (logits, spt_r)
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q = net(feat_metrix_meta, qry_edg,  fast_weights, bn_training=True)
                # logits_q = net(feat_metrix_meta, qry_edg,  fast_weights, bn_training=False)
                # logits_q = net(feat_metrix_meta, qry_edg,  fast_weights, bn_training=True, dropout_rate=0)#drop
                loss_q = loss_func(logits_q, qry_r)
                loss_mean += loss_q.detach().cpu().numpy()
                with torch.no_grad():
                    pred_score = F.softmax(logits_q, dim=1)
                    pred_score_list.append(pred_score)
                    y_truth_list.append(qry_r)
                    pred_q = pred_score.argmax(dim=1)
                    correct = torch.eq(pred_q, qry_r).sum().item() 
                    corrects[k + 1] = corrects[k + 1] + correct
                    qry_sz[k + 1] = qry_sz[k + 1] + len(qry_r)
        del net
        accs_test = np.array(corrects[-1]) / qry_sz[-1]        
        return accs_test, pred_score_list, y_truth_list,loss_mean



#***************************************************************************************************
df_drug_raw = pd.read_csv('df_570.csv')
df_extraction = pd.read_csv('df_interaction_570.csv')
mechanism = df_extraction['mechanism']
action = df_extraction['action']
drugA = df_extraction['drugA']
drugB = df_extraction['drugB']

df_drug_raw.to_csv("drug_raw.csv")
df_drug, pd_ddi = rename_ddi(df_drug_raw, mechanism, action, drugA, drugB)

y_true = np.array([])
y_score = np.zeros((0, 47), dtype=float)
y_pred = np.array([])
for idx in [0, 1, 2, 3, 4]:
    print("idx:", idx)
    data = Data_triplets(idx)   # seen[0] unseen[0]
    data.get_drug_related_triplets_dict()
    train_drug_dict_pool = data.train_drug_dict_pool
    test_drug_dict_pool = data.test_drug_dict_pool
    train_set_triplets = data.train_set_triplets
    test_set_triplets = data.test_set_triplets
    feature_sel = [0,1,2,3,4]
    maml = Meta().to(device)
    drug_pool_train= torch.tensor(list(train_drug_dict_pool.keys()))
    drug_pool_test= torch.tensor(list(test_drug_dict_pool.keys()))
    batcsz = 1
    pred_score_feature_best = []
    y_truth_feature_best = []
    for f in feature_sel:  
        print("feature:", f)
        maml.load_state_dict(torch.load("./ckpt/"+"570covgelucv01234k50adambatch1-"+str(idx)+"withrealfusion-"+str(f)+"best.pt"))
        if f == 0:
            feat_mtrx_sel = np.load("targetnew570.npy")
        if f == 1:
            feat_mtrx_sel = np.load("enzymenew570.npy")
        if f == 2:
            feat_mtrx_sel = np.load("pathwaynew570.npy")
        if f == 3:
            feat_mtrx_sel = np.load("smilenew570.npy")
        if f == 4:
            feat_mtrx_sel = (np.load("smilenew570.npy") + np.load("targetnew570.npy") + np.load("pathwaynew570.npy") + np.load("enzymenew570.npy")) / 4.0
        acc, pred_score_l_best, y_truth_l, l = maml.finetune(drug_pool_test.tolist(), feat_mtrx_sel)
        pred_score_feature_best.append(torch.cat(pred_score_l_best, 0))
        y_truth_feature_best.append(torch.cat(y_truth_l, 0))
    if len(pred_score_feature_best) == 5:
        pred_score_res = ((pred_score_feature_best[0] + pred_score_feature_best[1] + pred_score_feature_best[2] + pred_score_feature_best[3]+ pred_score_feature_best[4]) / 5).cpu().numpy()
    if len(pred_score_feature_best) == 4:
        pred_score_res = ((pred_score_feature_best[0] + pred_score_feature_best[1] + pred_score_feature_best[2] + pred_score_feature_best[3]) / 4).cpu().numpy()
    if len(pred_score_feature_best) == 3:
        pred_score_res = ((pred_score_feature_best[0] + pred_score_feature_best[1] + pred_score_feature_best[2]) / 3).cpu().numpy() 
    if len(pred_score_feature_best) == 1:
        pred_score_res = pred_score_feature_best[0].cpu().numpy() 
    pred_type = np.argmax(pred_score_res, axis=1)
    y_truth = y_truth_feature_best[0].cpu().numpy()
    cv_all_result, cv_each_result=evaluate(pred_type, pred_score_res, y_truth, 47, "f")
    print(cv_all_result)
    save_result(str(idx) + "570task2covgelucv01234k50adambatch1", 'all', "META", cv_all_result)
    save_result(str(idx) + "570task2covgelucv01234k50adambatch1", 'each', "META", cv_each_result)
    y_true = np.hstack((y_true, y_truth))
    y_pred = np.hstack((y_pred, pred_type))
    y_score = np.row_stack((y_score, pred_score_res))


all_result, each_result=evaluate(y_pred, y_score, y_true, 47, "f")
save_result("570task2covgelucv01234k50adambatch1", 'all', "META", all_result)
save_result("570task2covgelucv01234k50adambatch1", 'each', "META", each_result)
