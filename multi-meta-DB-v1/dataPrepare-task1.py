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
from torch_geometric.nn import GATConv
import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from pytorchtools import EarlyStopping
# from pytorchtools import BalancedDataParallel
# from radam import RAdam
import torch.nn.functional as F
from copy import deepcopy
from myModel import *
from myModel_GAT import *
random.seed(0)
device = torch.device('cuda')
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

def evaluate(pred_type, pred_score, y_test, event_num, set_name):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, classes=np.arange(event_num))

    precision, recall, th = multiclass_precision_recall_curve(y_one_hot, pred_score)

    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = precision_score(y_test, pred_type, average='micro')
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
#         result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
#                                           average=None)
#         result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
#                                          average=None)
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
def feature_vector(feature_name, df):  #每个通道特征计算
    # df are the 572 kinds of drugs
    # Jaccard Similarity
    def Jaccard(matrix):
        matrix = np.mat(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
        return numerator / denominator

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()    ##这个应该是通道 feature的原始数据，比如所有drug的enzyme
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1
    sim_matrix = Jaccard(np.array(df_feature))

    sim_matrix1 = np.array(sim_matrix)
    return sim_matrix1


def creat_drug_expres(df_drug):
    d_feature = {}
    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = []
    return d_feature 


def get_feature_dict(df_drug):
    feature_all = ["enzyme", "target", "pathway", "smile"]
    d_feature = creat_drug_expres(df_drug)      #创建每个药物名称对应空列表的字典
#     print("feature_dict:", d_feature )  #空列表
    for i in feature_all:
        vector_temp =  feature_vector(i, df_drug)
#        print("vector_temp",vector_temp)
#        print("d_feature:", d_feature)
        for j in range(len(df_drug['name'])):
            d_feature[df_drug['name'][j]].append(vector_temp[j])   #save features' all modal feature
#     print("d_feature:", d_feature)
            
    with open("d_feature_dict.pkl", "wb") as tf:           #每个药物对应的所有通道的特征，药物名称为字典的key，value为 n(feature_all) * vector size
        pickle.dump(d_feature, tf)
    print("Feature 学习并写入成功！！！")

def rename_ddi(df_drug, mechanism, action, drugA, drugB):  #药物、event重命名，生成ddi文件
    name_dict = {}

    label_dict = {}
    count={}
    # Transfrom the interaction event to number
    # Splice the features
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
#     print("label_dict:", label_dict)
    
    for i in range(len(d_event)):
            new_label.append(label_dict[d_event[i]])
#     print("new_label:", new_label)    #label rename as number
    
    for i in range(len(df_drug["name"])):
        name_dict[df_drug["name"][i]] = i + 1   #药物名称与字典对应

#     with open("drug_rename_dict.pkl", "wb") as tf:           #每个药物对应的所有通道的特征，药物名称为字典的key，value为 n(feature_all) * vector size
#         pickle.dump(name_dict, tf)
#     print("Rename_dict 学习并写入成功！！！")


    name_drugA_new = []
    name_drugB_new = []
    name_dfdrug_new = []
    for i in range(len(drugA)):
        name_drugA_new.append(name_dict[drugA[i]])
        name_drugB_new.append(name_dict[drugB[i]])
    for j in range(len(df_drug["name"])):
        name_dfdrug_new.append(name_dict[df_drug["name"][j]])
    
    df_drug["name"] = name_dfdrug_new
    df_drug.to_csv("df_drug.csv")
        
    pd_DDI = pd.DataFrame({"d1":name_drugA_new ,"type":new_label, "d2":name_drugB_new})
    pd_DDI.to_csv("ddis.csv")

    
    return df_drug, pd_DDI

    

def prepare(df_drug, feature_sel):  

    ########需要什么拿什么出来，构建特征矩阵#########
    vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)
    with open("d_feature_dict.pkl", "rb") as tf:
        feature_dict = pickle.load(tf)  

#     print("len(feature_dict):", len(feature_dict))
    for i in feature_sel:
        vector1 = []
        for j in range(len(df_drug["name"])):
#             print("vector1:",vector1)
            vector1.append(feature_dict[df_drug["name"][j]][i])
        vector = np.hstack((vector,vector1))   #需要参与训练的特征矩阵
#     print("feature_mtx.shape:", vector.shape)
    return vector     #feature_matrix


def Data_triplets(idx):  
    pd_ddi = pd.read_csv("./ddis.csv", usecols=[1, 2, 3], names=None)
    print(pd_ddi)
    ddi_rel_triplets_list = pd_ddi.values.tolist()
    print(ddi_rel_triplets_list[:10])
    print(len(ddi_rel_triplets_list))
    index_all_class = np.load("index_all_class.npy", allow_pickle = True)
    train_index = np.where(index_all_class != idx)
    test_index = np.where(index_all_class == idx)
    print(train_index[0].shape)
    print(test_index[0].shape)
    list_train = list(train_index[0])
    list_test = list(test_index[0])
    train_drug = {}
    test_drug = {}
    for i in list_train:
        drug1 = ddi_rel_triplets_list[i][0]
        drug2 = ddi_rel_triplets_list[i][2]
        if drug1 in train_drug.keys():
            train_drug[drug1]+=1
        else:
            train_drug[drug1]=1
        if drug2 in train_drug.keys():
            train_drug[drug2]+=1
        else:
            train_drug[drug2]=1
    for i in list_test:
        drug1 = ddi_rel_triplets_list[i][0]
        drug2 = ddi_rel_triplets_list[i][2]
        if drug1 in test_drug.keys():
            test_drug[drug1]+=1
        else:
            test_drug[drug1]=1
        if drug2 in test_drug.keys():
            test_drug[drug2]+=1
        else:
            test_drug[drug2]=1    
    print(train_drug.keys())
    print(test_drug.keys())
    out_drug = []
    for i in test_drug.keys():
        if i not in train_drug.keys():
            out_drug.append(i)
    print(out_drug)
    list_test_real = []
    for i in list_test:
        drug1 = ddi_rel_triplets_list[i][0]
        drug2 = ddi_rel_triplets_list[i][2]
        if(drug1 not in out_drug and drug2 not in out_drug):
            list_test_real.append(i)
    print(len(list_test_real))
    train_triplets = []
    test_triplets = []
    for i in list_train:
        train_triplets.append(ddi_rel_triplets_list[i])
    for i in list_test:
        test_triplets.append(ddi_rel_triplets_list[i])
    edge_index = np.zeros((2, 2 * len(train_triplets)),dtype = int)
    train_label = np.zeros(2 * len(train_triplets),dtype = int)
    j = 0
    for tri in train_triplets:
        edge_index[0][j] = tri[0] - 1
        edge_index[1][j+len(train_triplets)] = tri[0] - 1
        edge_index[1][j] = tri[2] - 1
        edge_index[0][j+len(train_triplets)] = tri[2] - 1
        train_label[j] = tri[1]
        train_label[j+len(train_triplets)] = tri[1]
        j+=1
    print(np.min(train_label))
    edge_index_test = np.zeros((2, 2*len(test_triplets)), dtype = int)
    test_label = np.zeros(2*len(test_triplets), dtype = int)
    j = 0
    for tri in test_triplets:
        edge_index_test[0][j] = tri[0] - 1
        edge_index_test[1][j+len(test_triplets)] = tri[0] - 1
        edge_index_test[1][j] = tri[2] - 1
        edge_index_test[0][j+len(test_triplets)] = tri[2] - 1
        test_label[j] = tri[1]
        test_label[j+len(test_triplets)] = tri[1]
        j+=1
    print(edge_index_test)
    print(np.min(test_label))
    return torch.Tensor(edge_index).type(torch.LongTensor).to(device),torch.Tensor(train_label).type(torch.LongTensor).to(device), torch.Tensor(edge_index_test).type(torch.LongTensor).to(device), torch.Tensor(test_label).type(torch.LongTensor).to(device)




def data_iter(batch_size):
    num = edge_index.shape[1]
    indices = list(range(num))
    random.shuffle(indices)
    for i in range(0, num, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num)])
        yield j

class GCN_Net(nn.Module):
    def __init__(self, num_inputs = 572, hidden_1 = 256, hidden_2 = 128, class_num = 65, n_heads = 2):  #n_heads = 1
        super(GCN_Net, self).__init__()  
        self.conv1 = GATConv(num_inputs, hidden_1, heads = n_heads, concat=False)
#         self.conv1 = GCNConv(num_inputs, hidden_1)
        self.bn1 =torch.nn.BatchNorm1d(hidden_1)
        self.conv2 = GATConv(hidden_1, hidden_1, heads = n_heads, concat=False)
#         self.conv1 = GCNConv(hidden_1, hidden_1)
        self.bn2 =torch.nn.BatchNorm1d(hidden_1)
        self.lin1 = nn.Linear(hidden_1*2, hidden_2)
        self.bn3 =torch.nn.BatchNorm1d(hidden_2)
        self.lin2 = nn.Linear(hidden_2, hidden_2)
        self.bn4 =torch.nn.BatchNorm1d(hidden_2)
        self.fc = nn.Linear(hidden_2, class_num)

    def forward(self, edge_index, x, edge_id):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x1 = x[edge_id[0]]
        x2 = x[edge_id[1]]
        x = torch.cat((x1, x2), dim = 1)
        x = self.lin1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.fc(x)
        return x
    
class Model_Task1(nn.Module):
    def __init__(self, ):
        super(Model_Task1, self).__init__()        
        self.lr = 0.001
#         self.bs = 1024
        self.bs = 512
        self.net = GCN_Net()
#         self.optim = optim.Adam(self.net.parameters(), lr = self.lr) # no decay 
        self.optim = optim.Adam(self.net.parameters(), lr = self.lr, weight_decay=0.00001) # no decay 
        self.loss_fn = nn.CrossEntropyLoss().to(device)
    def forward(self, edge_index_train, train_label, feature):
        model.train()
        for i, batch in enumerate(data_iter(self.bs)):
            out = self.net(edge_index_train, feature, edge_index_train[:, batch])
            label = train_label[batch]    
            loss = self.loss_fn(out, label)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
#         out = self.net(edge_index_train, feature, edge_index_train)
#         label = train_label      
#         loss = self.loss_fn(out, label)
#         self.optim.zero_grad()
#         loss.backward()
#         self.optim.step()
#         accracy = np.mean((torch.argmax(out,1)==label).cpu().numpy())
        return 1.0
    def predict(self, edge_index_train, edge_index_test, test_label, feature):
        model.eval()
        out = self.net(edge_index_train, feature, edge_index_test)
        loss = self.loss_fn(out, test_label)
        score = F.softmax(out, dim=1)
        accracy = np.mean((torch.argmax(score,1)==test_label).cpu().numpy())
        return accracy, score, test_label, loss
        

#***************************************************************************************************
conn = sqlite3.connect("./event.db")
df_drug_raw = pd.read_sql('select * from drug;', conn)
df_event = pd.read_sql('select * from event_number;', conn)
df_extraction = pd.read_sql('select * from extraction;', conn)
mechanism = df_extraction['mechanism']
action = df_extraction['action']
drugA = df_extraction['drugA']
drugB = df_extraction['drugB']

df_drug_raw.to_csv("drug_raw.csv")
df_drug, pd_ddi = rename_ddi(df_drug_raw, mechanism,action,drugA,drugB)






# data = Data_triplets(4) 

# drug_root= 542
# print("train_drug_dict_pool:", train_drug_dict_pool)
# print("drug_root:", drug_root)
# feat_metrix_meta, spt_edg, spt_r, qry_edg, qry_r = generate_task_data_train(drug_root,train_drug_dict_pool,test_drug_dict_pool,train_set_triplets,test_set_triplets,feat_mtrx_sel,True)
# print("feat_metrix_meta:", feat_metrix_meta)

y_true = np.array([])
y_score = np.zeros((0, 65), dtype=float)
y_pred = np.array([])
for idx in range(5):
    edge_index, train_label, edge_index_test, test_label = Data_triplets(idx)   # seen[0] unseen[0]
#     feature_sel = [0,1,2,3]
#     feature_sel = [0, 1, 2, 4]
    feature_sel = [4, 0, 1, 2]

#     feature_sel = [4]

    model = Model_Task1().to(device)  #94
    pred_score_feature_best = []
    y_truth_feature_best = []
    for i in feature_sel: 
#         model = Model_Task1().to(device)
        if i == 4:
            feat_mtrx_sel = np.load("5000Smile_cluster_572.npy")
#             feat_mtrx_sel = np.load("morgen_smile_new.npy")
        else:
            feat_mtrx_sel = prepare(df_drug, [i])
        feat_mtrx_sel = torch.Tensor(feat_mtrx_sel).to(device)
        epochs = 200
        best_acc = -1
        best_loss = 100000
        for epoch in range(epochs):
            acc = model(edge_index, train_label, feat_mtrx_sel)
#             print("epoch, acc:",epoch, acc)
            if epoch % 5 ==0:
                acc, pred_score_l, y_truth_l, loss = model.predict(edge_index, edge_index_test, test_label, feat_mtrx_sel)
                loss = loss.detach().cpu().numpy()
                print("epoch,loss:", epoch, loss)
                if(acc > best_acc):
                    best_acc = acc
                    pred_score_l_best = pred_score_l
#                 if(loss < best_loss):
#                     best_loss = loss
#                     pred_score_l_best = pred_score_l
                print_file("epoch: {}, acc:{}, best_acc: {}".format(epoch, acc, best_acc), "./result.out")
        pred_score_feature_best.append(pred_score_l_best)
        y_truth_feature_best.append(y_truth_l)
    if len(pred_score_feature_best) == 4:
        pred_score_res = ((pred_score_feature_best[0] + pred_score_feature_best[1] + pred_score_feature_best[2] + pred_score_feature_best[3]) / 4).detach().cpu().numpy()
    if len(pred_score_feature_best) == 3:
        pred_score_res = ((pred_score_feature_best[0] + pred_score_feature_best[1] + pred_score_feature_best[2]) / 3).detach().cpu().numpy()
    if len(pred_score_feature_best) == 1:
        pred_score_res = pred_score_feature_best[0].detach().cpu().numpy()
    pred_type = np.argmax(pred_score_res, axis=1)
    y_truth = y_truth_feature_best[0].detach().cpu().numpy()
    cv_all_result, cv_each_result=evaluate(pred_type, pred_score_res, y_truth, 65, "f")
    print(cv_all_result)
    y_true = np.hstack((y_true, y_truth))
    y_pred = np.hstack((y_pred, pred_type))
    y_score = np.row_stack((y_score, pred_score_res))


all_result, each_result=evaluate(y_pred, y_score, y_true, 65, "f")
save_result("target_enz_5_150_enz_traget_4012smile_withdecay_bs512_epoch200", 'all', "META", all_result)
save_result("target_enz_5_150_enz_traget_4012smile_withdecay_bs512_epoch200", 'each', "META", each_result)


# edge_index, train_label, edge_index_test, test_label = Data_triplets(0)
