# -*- coding: utf-8 -*-
from numpy.random import seed
import csv
import numpy as np
import random
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from torch_geometric.nn import GATConv
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
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
#     result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
#     result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
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
    df_drug.to_csv("df_drug397.csv")
    pd_DDI = pd.DataFrame({"d1":name_drugA_new ,"type":new_label, "d2":name_drugB_new})
    pd_DDI.to_csv("ddis.csv")
    return df_drug, pd_DDI

def Data_triplets(idx): 
    pd_ddi = pd.read_csv("./ddis.csv", usecols=[1, 2, 3], names=None)
    ddi_rel_triplets_list = pd_ddi.values.tolist()  #(drugA, type, drugB)      
    seen = np.load("389seen_rename.npy", allow_pickle = True)
    unseen = np.load("389unseen_rename.npy", allow_pickle = True)
    seen_drugs = seen[idx]
    unseen_drugs = unseen[idx]
    train_triplets = []
    test_triplets = []
    others_triplets = []
    for l2 in ddi_rel_triplets_list:
        if l2[0] in seen_drugs and l2[2] in seen_drugs:
            train_triplets.append(l2)
        elif (l2[0] in seen_drugs and l2[2] in unseen_drugs) or (l2[0] in unseen_drugs and l2[2] in seen_drugs):
            test_triplets.append(l2)
        else:
            others_triplets.append(l2)
    print("len(test_triplets):", len(test_triplets))
    print("len(train_triplets):", len(train_triplets))
    edge_index = np.zeros((2, 2 * len(train_triplets)),dtype = int)
    train_label = np.zeros(2 * len(train_triplets),dtype = int)
    j = 0
    for tri in train_triplets:
        edge_index[0][j] = tri[0]
        edge_index[1][j+len(train_triplets)] = tri[0]
        edge_index[1][j] = tri[2]
        edge_index[0][j+len(train_triplets)] = tri[2]
        train_label[j] = tri[1]
        train_label[j+len(train_triplets)] = tri[1]
        j+=1
    print(np.min(train_label))
    edge_index_test = np.zeros((2, 2*len(test_triplets)), dtype = int)
    test_label = np.zeros(2*len(test_triplets), dtype = int)
    j = 0
    for tri in test_triplets:
        edge_index_test[0][j] = tri[0]
        edge_index_test[1][j+len(test_triplets)] = tri[0]
        edge_index_test[1][j] = tri[2]
        edge_index_test[0][j+len(test_triplets)] = tri[2]
        test_label[j] = tri[1]
        test_label[j+len(test_triplets)] = tri[1]
        j+=1
    print("edge_index:", edge_index.shape)
    print("edge_index_test:", edge_index_test.shape)
    print(np.min(test_label))
    edge_index_train_all = np.concatenate([edge_index,edge_index_test],axis=1)
    print("edge_index_train_all:", edge_index_train_all.shape)
    return torch.Tensor(edge_index).type(torch.LongTensor).to(device),torch.Tensor(train_label).type(torch.LongTensor).to(device), torch.Tensor(edge_index_test).type(torch.LongTensor).to(device), torch.Tensor(test_label).type(torch.LongTensor).to(device), torch.Tensor(edge_index_train_all).type(torch.LongTensor).to(device)


def data_iter(batch_size):
    num = edge_index.shape[1]
    indices = list(range(num))
    random.shuffle(indices)
    for i in range(0, num, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num)])
        yield j

class GCN_Net_gelu(nn.Module):
    def __init__(self, num_inputs = 389, hidden_1 = 256, hidden_2 = 128, class_num = 52, n_heads = 2):  #n_heads = 1
        super(GCN_Net_gelu, self).__init__()  
        # self.conv1 = GATConv(num_inputs, hidden_1, heads = n_heads, concat=False)
        self.conv1 = GCNConv(num_inputs, hidden_1)
        self.bn1 =torch.nn.BatchNorm1d(hidden_1)
        # self.conv2 = GATConv(hidden_1, hidden_1, heads = n_heads, concat=False)
        self.conv2 = GCNConv(hidden_1, hidden_1)
        self.bn2 =torch.nn.BatchNorm1d(hidden_1)
        self.lin1 = nn.Linear(hidden_1*2, hidden_2)
        self.bn3 =torch.nn.BatchNorm1d(hidden_2)
        self.lin2 = nn.Linear(hidden_2, hidden_2)
        self.bn4 =torch.nn.BatchNorm1d(hidden_2)
        self.fc = nn.Linear(hidden_2, class_num)

    def forward(self, edge_index, x, edge_id):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.gelu(x)
        x1 = x[edge_id[0]]
        x2 = x[edge_id[1]]
        x = torch.cat((x1, x2), dim = 1)
        x = self.lin1(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.lin2(x)
        x = self.bn4(x)
        x = F.gelu(x)
        x = self.fc(x)
        return x
    
class Model_Task1(nn.Module):
    def __init__(self, ):
        super(Model_Task1, self).__init__()        
        self.lr = 0.001
#         self.bs = 1024
        self.bs = 512
        self.net = GCN_Net_gelu()
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
        return 1.0
    def predict(self, edge_index_train_all, edge_index_test, test_label, feature):
        model.eval()
        out = self.net(edge_index_train_all, feature, edge_index_test)
        loss = self.loss_fn(out, test_label)
        score = F.softmax(out, dim=1)
        accracy = np.mean((torch.argmax(score,1)==test_label).cpu().numpy())
        return accracy, score, test_label, loss
        

#***************************************************************************************************
df_drug_raw = pd.read_csv('df_389.csv')
df_extraction = pd.read_csv('df_interaction_389.csv')
mechanism = df_extraction['mechanism']
action = df_extraction['action']
drugA = df_extraction['drugA']
drugB = df_extraction['drugB']

df_drug_raw.to_csv("drug_raw.csv")
df_drug, pd_ddi = rename_ddi(df_drug_raw, mechanism, action, drugA, drugB)
y_true = np.array([])
y_score = np.zeros((0, 52), dtype=float)
y_pred = np.array([])
for idx in range(5):
    edge_index, train_label, edge_index_test, test_label, edge_index_train_all  = Data_triplets(idx)
    print("edge_index:", edge_index.shape)
    print("edge_index_train_all:", edge_index_train_all.shape)
    feature_sel = [0, 1, 2, 3, 4, 5]
    # model = Model_Task1().to(device)  #94
    pred_score_feature_best = []
    y_truth_feature_best = []
    for f in feature_sel: 
        model = Model_Task1().to(device)  #94
        if f == 0:
            feat_mtrx_sel = np.load("targetnew389.npy")
        if f == 1:
            feat_mtrx_sel = np.load("enzymenew389.npy")
        if f == 2:
            feat_mtrx_sel = np.load("enzymenew389.npy")
        if f == 3:
            feat_mtrx_sel = np.load("smilenew389.npy")
        if f == 4:
            feat_mtrx_sel = np.load("diseasenew389.npy")
        if f == 5:
            feat_mtrx_sel = np.load("genenew389.npy")
        feat_mtrx_sel = torch.Tensor(feat_mtrx_sel).to(device)
        epochs = 200
        best_acc = -1
        best_loss = 100000
        for epoch in range(epochs):
            acc = model(edge_index, train_label, feat_mtrx_sel)
            if epoch % 5 ==0:
                acc, pred_score_l, y_truth_l, loss= model.predict(edge_index_train_all, edge_index_test, test_label, feat_mtrx_sel)
                loss = loss.detach().cpu().numpy()
                print("epoch,loss:", epoch, loss)
                if(acc > best_acc):
                    best_acc = acc
                    pred_score_l_best = pred_score_l
                print_file("task1 epoch: {}, acc:{}, best_acc: {}".format(epoch, acc, best_acc), "./result.out")
        pred_score_feature_best.append(pred_score_l_best)
        y_truth_feature_best.append(y_truth_l)
    if len(pred_score_feature_best) == 6:
        pred_score_res = ((pred_score_feature_best[0] + pred_score_feature_best[1] + pred_score_feature_best[2] + pred_score_feature_best[3]+ pred_score_feature_best[4] + pred_score_feature_best[5]) / 6).detach().cpu().numpy()
    if len(pred_score_feature_best) == 4:
        pred_score_res = ((pred_score_feature_best[0] + pred_score_feature_best[1] + pred_score_feature_best[2] + pred_score_feature_best[3]) / 4).detach().cpu().numpy()
    if len(pred_score_feature_best) == 3:
        pred_score_res = ((pred_score_feature_best[0] + pred_score_feature_best[1] + pred_score_feature_best[2]) / 3).detach().cpu().numpy()
    if len(pred_score_feature_best) == 1:
        pred_score_res = pred_score_feature_best[0].detach().cpu().numpy()
    pred_type = np.argmax(pred_score_res, axis=1)
    y_truth = y_truth_feature_best[0].detach().cpu().numpy()
    # cv_all_result, cv_each_result=evaluate(pred_type, pred_score_res, y_truth, 52, "f")
    # print(cv_all_result)
    y_true = np.hstack((y_true, y_truth))
    y_pred = np.hstack((y_pred, pred_type))
    y_score = np.row_stack((y_score, pred_score_res))


all_result, each_result=evaluate(y_pred, y_score, y_true, 52, "f")
save_result("task2gelugcn_nogamtenhance", 'all', "META", all_result)
save_result("task2gelugcn_nogamtenhance", 'each', "META", each_result)
