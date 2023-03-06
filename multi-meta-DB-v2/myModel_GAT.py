import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv,SAGPooling,RGCNConv,global_add_pool
from torch_geometric.data import Data
from mygcn import *
import numpy as np
import random

class Mymodel_GAT(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens, num_edg_hiddens):
        super(Mymodel_GAT, self).__init__()
        
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.num_edg_hiddens = num_edg_hiddens
        self.heads = 4
        ## layer0, gat_my1, var[0, 1, 2, 3]  w0,b0,attr_src, att_dst
        self.gat_my1 = GATConv_my(self.num_inputs, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w0 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_inputs]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w0)
        self.vars.append(w0) #w
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) #b
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        #layer1, bn, var[4,5] , vars_bn[0,1]   w1,b1
        w1 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        #layer2 relu 
        #layer3 dropout
        
        
        #layer 4 gcn_my2  var[6,7,8,9]   w2,b2,src,dst
        self.gat_my2 = GATConv_my(self.num_hiddens, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w2 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_hiddens]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) 
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        
        #layer 5 bn    nvar[10,11]    vars_bn[2,3]
        w3 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w3)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        #layer 6 gelu
        
        
        #******* edge feature ####
        
        #layer 7 linear var[12,13]
        w4 = nn.Parameter(torch.ones([self.num_edg_hiddens,  self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w4)
        self.vars.append(w4)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))             
        # layer8 bn       var[14,15]  vars_bn[4,5]
        w5 = nn.Parameter(torch.ones(self.num_edg_hiddens))    # [ch_out]      
        self.vars.append(w5)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_edg_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_edg_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var])       
        
        #layer 9 relu
        #layer10 linear var[16,17]
        w6 = nn.Parameter(torch.ones([self.num_edg_hiddens,  self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w6)
        self.vars.append(w6)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))   
        
        
        # layer11 bn       var[18,19]  vars_bn[6,7]
        w7 = nn.Parameter(torch.ones(self.num_edg_hiddens))    # [ch_out]      
        self.vars.append(w7)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_edg_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_edg_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var])              
                          
        #layer 12 linear var[20,21]
        w8 = nn.Parameter(torch.ones([self.num_outputs, self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w8)
        self.vars.append(w8)
        self.vars.append(nn.Parameter(torch.zeros(self.num_outputs)))
        
        
    def forward(self, x, edge_index, vars = None, bn_training = True):
        
        if vars is None:
            vars = self.vars       
            
        w0, b0, att_src, att_dst = vars[0], vars[1], vars[2], vars[3]
#         print("edge_index.shape:", edge_index.shape)
        x = self.gat_my1(x, edge_index, w0, b0, att_src, att_dst)   # layer 0 gcn_my1
        
        w1, b1 = vars[4], vars[5]                
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]  
        x = F.batch_norm(x, running_mean, running_var, weight=w1, bias=b1, training=bn_training)   #layer 1 bn
#         print("x0.size:", x.size())
#         print("x.size:", x.size())
#         x1 = F.gelu(x)    #layer2  bn
        x1 = F.relu(x)    #layer2  bn
        
        
        w2, b2, att_src1, att_dst1 =vars[6], vars[7],vars[8], vars[9]
        x = self.gat_my2(x1, edge_index, w2, b2, att_src1, att_dst1)  #layer 4 gcn_my2
        w3, b3 = vars[10], vars[11]
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=w3, bias=b3, training=bn_training)   #layer5 bn
#         x = F.gelu(x + x1)     #layer6 gelu
        x = F.relu(x + x1)     #layer6 gelu
        
        #### Get edge features
        edge_src = x[edge_index[0]]
        edge_dst = x[edge_index[1]]
#         print("edge_src:",edge_src)
#         print("edge_dst:",edge_dst)

        edge_feat = torch.cat((edge_src,edge_dst),axis = 1)
         
        w4, b4 = vars[12], vars[13]     
        edge_feat = F.linear(edge_feat, w4, b4)   # layer 7 linear bn relu
        w5, b5 = vars[14], vars[15]
        running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
        edge_feat = F.batch_norm(edge_feat, running_mean, running_var, weight=w5, bias=b5, training=bn_training)  # layer 8 bn
#         edge_feat = F.gelu(edge_feat)       #layer9 relu
        edge_feat = F.relu(edge_feat)       #layer9 relu

        w6, b6 = vars[16], vars[17]     
        edge_feat = F.linear(edge_feat, w6, b6)   # layer 7 linear bn relu
        w7, b7 = vars[18], vars[19]
        running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
        edge_feat = F.batch_norm(edge_feat, running_mean, running_var, weight=w7, bias=b7, training=bn_training)  # layer 8 bn
#         edge_feat = F.gelu(edge_feat)       #layer9 relu
        edge_feat = F.relu(edge_feat)       #layer9 relu
        
        w8, b8 = vars[20], vars[21]
        rel_pre = F.linear(edge_feat, w8, b8)  #layer 10 linear
        return rel_pre
    def zero_grad(self, vars=None): 
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars 
    
    
    
    
    
class Mymodel_GAT_cov(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens, num_edg_hiddens):
        super(Mymodel_GAT_cov, self).__init__()
        
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.num_edg_hiddens = num_edg_hiddens
        self.cov2KerSize = 50
        #self.cov2KerSize = 16
        self.heads = 4
        
        
        ## layer0, gat_my1, var[0, 1, 2, 3]  w0,b0,attr_src, att_dst
        self.gat_my1 = GATConv_my(self.num_inputs, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w0 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_inputs]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w0)
        self.vars.append(w0) #w
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) #b
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        #layer1, bn, var[4,5] , vars_bn[0,1]   w1,b1
        w1 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        
        #layer 4 gcn_my2  var[6,7,8,9]   w2,b2,src,dst
        self.gat_my2 = GATConv_my(self.num_hiddens, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w2 = nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_hiddens]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) 
        self.vars.append(torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append(torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        #layer 5 bn    nvar[10,11]    vars_bn[2,3]
        w3 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w3)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        
        #******* edge feature ####
        #layer 7 cov2 var[12,13]
        w4 = nn.Parameter(torch.ones([1,1,2,self.cov2KerSize]))
        torch.nn.init.kaiming_normal_(w4)
        self.vars.append(w4)
        self.vars.append(nn.Parameter(torch.zeros(1)))   
        
        #layer10 cov1 var[14,15]
        w6 = nn.Parameter(torch.ones([1,1,self.cov2KerSize]))
        torch.nn.init.kaiming_normal_(w6)
        self.vars.append(w6)
        self.vars.append(nn.Parameter(torch.zeros(1)))             
    
    
        #layer 12 linear var[16,17]
        w8 = nn.Parameter(torch.ones([self.num_outputs, 512]))
        torch.nn.init.kaiming_normal_(w8)
        self.vars.append(w8)
        self.vars.append(nn.Parameter(torch.zeros(self.num_outputs)))
        
        
    def forward(self, x, edge_index, vars = None, bn_training = True):
        
        if vars is None:
            vars = self.vars       
            
        w0, b0, att_src, att_dst = vars[0], vars[1], vars[2], vars[3]
#         print("edge_index.shape:", edge_index.shape)
        x = self.gat_my1(x, edge_index, w0, b0, att_src, att_dst)   # layer 0 gcn_my1
        
        w1, b1 = vars[4], vars[5]                
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]  
        x = F.batch_norm(x, running_mean, running_var, weight=w1, bias=b1, training=bn_training)   #layer 1 bn
        x1 = F.relu(x)    #layer2  bn
        
        
        w2, b2, att_src1, att_dst1 =vars[6], vars[7],vars[8], vars[9]
        x = self.gat_my2(x1, edge_index, w2, b2, att_src1, att_dst1)  #layer 4 gcn_my2
        w3, b3 = vars[10], vars[11]
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=w3, bias=b3, training=bn_training)   #layer5 bn
        x = F.relu(x)     #layer6 gelu
        
        #### Get edge features
        edge_src = x[edge_index[0]]
        edge_dst = x[edge_index[1]]
        edge_feat = torch.cat((edge_src,edge_dst),axis = 1)
         
        w4, b4 = vars[12], vars[13]     
        edge_feat=edge_feat.view(-1,1,2,self.num_hiddens)
        edge_feat=F.relu(F.conv2d(edge_feat, w4, b4))
        edge_feat=edge_feat.view(-1,self.num_hiddens-self.cov2KerSize+1, 1)  
        edge_feat=edge_feat.permute(0,2,1)
        w5, b5 = vars[14], vars[15] 
        edge_feat=F.relu(F.conv1d(edge_feat,w5,b5))
        edge_feat=F.adaptive_avg_pool1d(edge_feat, 512)
        edge_feat=edge_feat.contiguous().view(-1, 512)        
        w8, b8 = vars[16], vars[17]
        rel_pre = F.linear(edge_feat, w8, b8)  #layer 10 linear
        return rel_pre
    def zero_grad(self, vars=None): 
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars 

class Mymodel_GAT_cov_residue(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens, num_edg_hiddens):
        super(Mymodel_GAT_cov_residue, self).__init__()
        
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.num_edg_hiddens = num_edg_hiddens
        self.cov2KerSize = 50
#         self.cov2KerSize = 32
        self.heads = 4
        
        
        ## layer0, gat_my1, var[0, 1, 2, 3]  w0,b0,attr_src, att_dst
        self.gat_my1 = GATConv_my(self.num_inputs, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w0 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_inputs]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w0)
        self.vars.append(w0) #w
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) #b
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        #layer1, bn, var[4,5] , vars_bn[0,1]   w1,b1
        w1 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        
        #layer 4 gcn_my2  var[6,7,8,9]   w2,b2,src,dst
        self.gat_my2 = GATConv_my(self.num_hiddens, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w2 = nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_hiddens]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) 
        self.vars.append(torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append(torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        #layer 5 bn    nvar[10,11]    vars_bn[2,3]
        w3 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w3)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        
        #******* edge feature ####
        #layer 7 cov2 var[12,13]
        w4 = nn.Parameter(torch.ones([1,1,2,self.cov2KerSize]))
        torch.nn.init.kaiming_normal_(w4)
        self.vars.append(w4)
        self.vars.append(nn.Parameter(torch.zeros(1)))   
        
        #layer10 cov1 var[14,15]
        w6 = nn.Parameter(torch.ones([1,1,self.cov2KerSize]))
        torch.nn.init.kaiming_normal_(w6)
        self.vars.append(w6)
        self.vars.append(nn.Parameter(torch.zeros(1)))             
    
    
        #layer 12 linear var[16,17]
        w8 = nn.Parameter(torch.ones([self.num_outputs, 512]))
        torch.nn.init.kaiming_normal_(w8)
        self.vars.append(w8)
        self.vars.append(nn.Parameter(torch.zeros(self.num_outputs)))
        
        
    def forward(self, x, edge_index, vars = None, bn_training = True):
        
        if vars is None:
            vars = self.vars       
            
        w0, b0, att_src, att_dst = vars[0], vars[1], vars[2], vars[3]
#         print("edge_index.shape:", edge_index.shape)
        x = self.gat_my1(x, edge_index, w0, b0, att_src, att_dst)   # layer 0 gcn_my1
        
        w1, b1 = vars[4], vars[5]                
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]  
        x = F.batch_norm(x, running_mean, running_var, weight=w1, bias=b1, training=bn_training)   #layer 1 bn
        x1 = F.relu(x)    #layer2  bn
        
        
        w2, b2, att_src1, att_dst1 =vars[6], vars[7],vars[8], vars[9]
        x = self.gat_my2(x1, edge_index, w2, b2, att_src1, att_dst1)  #layer 4 gcn_my2
        w3, b3 = vars[10], vars[11]
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=w3, bias=b3, training=bn_training)   #layer5 bn
        x = F.relu(x + x1)     #layer6 gelu
        
        #### Get edge features
        edge_src = x[edge_index[0]]
        edge_dst = x[edge_index[1]]
        edge_feat = torch.cat((edge_src,edge_dst),axis = 1)
         
        w4, b4 = vars[12], vars[13]     
        edge_feat=edge_feat.view(-1,1,2,self.num_hiddens)
        edge_feat=F.relu(F.conv2d(edge_feat, w4, b4))
        edge_feat=edge_feat.view(-1,self.num_hiddens-self.cov2KerSize+1, 1)  
        edge_feat=edge_feat.permute(0,2,1)
        w5, b5 = vars[14], vars[15] 
        edge_feat=F.relu(F.conv1d(edge_feat,w5,b5))
        edge_feat=F.adaptive_avg_pool1d(edge_feat, 512)
        edge_feat=edge_feat.contiguous().view(-1, 512)        
        w8, b8 = vars[16], vars[17]
        rel_pre = F.linear(edge_feat, w8, b8)  #layer 10 linear
        return rel_pre
    def zero_grad(self, vars=None): 
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars 
    
class Mymodel_GAT_cov_residue_gelu(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens, num_edg_hiddens):
        super(Mymodel_GAT_cov_residue_gelu, self).__init__()
        
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.num_edg_hiddens = num_edg_hiddens
        self.cov2KerSize = 50
#         self.cov2KerSize = 32
        self.heads = 4
        
        
        ## layer0, gat_my1, var[0, 1, 2, 3]  w0,b0,attr_src, att_dst
        self.gat_my1 = GATConv_my(self.num_inputs, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w0 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_inputs]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w0)
        self.vars.append(w0) #w
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) #b
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        #layer1, bn, var[4,5] , vars_bn[0,1]   w1,b1
        w1 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        
        #layer 4 gcn_my2  var[6,7,8,9]   w2,b2,src,dst
        self.gat_my2 = GATConv_my(self.num_hiddens, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w2 = nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_hiddens]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) 
        self.vars.append(torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append(torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        #layer 5 bn    nvar[10,11]    vars_bn[2,3]
        w3 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w3)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        
        #******* edge feature ####
        #layer 7 cov2 var[12,13]
        w4 = nn.Parameter(torch.ones([1,1,2,self.cov2KerSize]))
        torch.nn.init.kaiming_normal_(w4)
        self.vars.append(w4)
        self.vars.append(nn.Parameter(torch.zeros(1)))   
        
        #layer10 cov1 var[14,15]
        w6 = nn.Parameter(torch.ones([1,1,self.cov2KerSize]))
        torch.nn.init.kaiming_normal_(w6)
        self.vars.append(w6)
        self.vars.append(nn.Parameter(torch.zeros(1)))             
    
    
        #layer 12 linear var[16,17]
        w8 = nn.Parameter(torch.ones([self.num_outputs, 512]))
        torch.nn.init.kaiming_normal_(w8)
        self.vars.append(w8)
        self.vars.append(nn.Parameter(torch.zeros(self.num_outputs)))
        
        
    def forward(self, x, edge_index, vars = None, bn_training = True):
        
        if vars is None:
            vars = self.vars       
            
        w0, b0, att_src, att_dst = vars[0], vars[1], vars[2], vars[3]
#         print("edge_index.shape:", edge_index.shape)
        x = self.gat_my1(x, edge_index, w0, b0, att_src, att_dst)   # layer 0 gcn_my1
        
        w1, b1 = vars[4], vars[5]                
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]  
        x = F.batch_norm(x, running_mean, running_var, weight=w1, bias=b1, training=bn_training)   #layer 1 bn
        x1 = F.gelu(x)    #layer2  bn
        
        
        w2, b2, att_src1, att_dst1 =vars[6], vars[7],vars[8], vars[9]
        x = self.gat_my2(x1, edge_index, w2, b2, att_src1, att_dst1)  #layer 4 gcn_my2
        w3, b3 = vars[10], vars[11]
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=w3, bias=b3, training=bn_training)   #layer5 bn
        x = F.gelu(x + x1)     #layer6 gelu
        
        #### Get edge features
        edge_src = x[edge_index[0]]
        edge_dst = x[edge_index[1]]
        edge_feat = torch.cat((edge_src,edge_dst),axis = 1)
         
        w4, b4 = vars[12], vars[13]     
        edge_feat=edge_feat.view(-1,1,2,self.num_hiddens)
        edge_feat=F.gelu(F.conv2d(edge_feat, w4, b4))
        edge_feat=edge_feat.view(-1,self.num_hiddens-self.cov2KerSize+1, 1)  
        edge_feat=edge_feat.permute(0,2,1)
        w5, b5 = vars[14], vars[15] 
        edge_feat=F.gelu(F.conv1d(edge_feat,w5,b5))
        edge_feat=F.adaptive_avg_pool1d(edge_feat, 512)
        edge_feat=edge_feat.contiguous().view(-1, 512)        
        w8, b8 = vars[16], vars[17]
        rel_pre = F.linear(edge_feat, w8, b8)  #layer 10 linear
        return rel_pre
    def zero_grad(self, vars=None): 
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars     
class Mymodel_GAT_cov_gelu(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens, num_edg_hiddens):
        super(Mymodel_GAT_cov_gelu, self).__init__()
        
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.num_edg_hiddens = num_edg_hiddens
        self.cov2KerSize = 50
#         self.cov2KerSize = 32
        self.heads = 4
        
        
        ## layer0, gat_my1, var[0, 1, 2, 3]  w0,b0,attr_src, att_dst
        self.gat_my1 = GATConv_my(self.num_inputs, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w0 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_inputs]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w0)
        self.vars.append(w0) #w
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) #b
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        #layer1, bn, var[4,5] , vars_bn[0,1]   w1,b1
        w1 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        
        #layer 4 gcn_my2  var[6,7,8,9]   w2,b2,src,dst
        self.gat_my2 = GATConv_my(self.num_hiddens, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w2 = nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_hiddens]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) 
        self.vars.append(torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append(torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        #layer 5 bn    nvar[10,11]    vars_bn[2,3]
        w3 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w3)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        
        #******* edge feature ####
        #layer 7 cov2 var[12,13]
        w4 = nn.Parameter(torch.ones([1,1,2,self.cov2KerSize]))
        torch.nn.init.kaiming_normal_(w4)
        self.vars.append(w4)
        self.vars.append(nn.Parameter(torch.zeros(1)))   
        
        #layer10 cov1 var[14,15]
        w6 = nn.Parameter(torch.ones([1,1,self.cov2KerSize]))
        torch.nn.init.kaiming_normal_(w6)
        self.vars.append(w6)
        self.vars.append(nn.Parameter(torch.zeros(1)))             
    
    
        #layer 12 linear var[16,17]
        w8 = nn.Parameter(torch.ones([self.num_outputs, 512]))
        torch.nn.init.kaiming_normal_(w8)
        self.vars.append(w8)
        self.vars.append(nn.Parameter(torch.zeros(self.num_outputs)))
        
        
    def forward(self, x, edge_index, vars = None, bn_training = True):
        
        if vars is None:
            vars = self.vars       
            
        w0, b0, att_src, att_dst = vars[0], vars[1], vars[2], vars[3]
#         print("edge_index.shape:", edge_index.shape)
        x = self.gat_my1(x, edge_index, w0, b0, att_src, att_dst)   # layer 0 gcn_my1
        
        w1, b1 = vars[4], vars[5]                
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]  
        x = F.batch_norm(x, running_mean, running_var, weight=w1, bias=b1, training=bn_training)   #layer 1 bn
        x1 = F.gelu(x)    #layer2  bn
        
        
        w2, b2, att_src1, att_dst1 =vars[6], vars[7],vars[8], vars[9]
        x = self.gat_my2(x1, edge_index, w2, b2, att_src1, att_dst1)  #layer 4 gcn_my2
        w3, b3 = vars[10], vars[11]
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=w3, bias=b3, training=bn_training)   #layer5 bn
        x = F.gelu(x)     #layer6 gelu
        
        #### Get edge features
        edge_src = x[edge_index[0]]
        edge_dst = x[edge_index[1]]
        edge_feat = torch.cat((edge_src,edge_dst),axis = 1)
         
        w4, b4 = vars[12], vars[13]     
        edge_feat=edge_feat.view(-1,1,2,self.num_hiddens)
        edge_feat=F.gelu(F.conv2d(edge_feat, w4, b4))
        edge_feat=edge_feat.view(-1,self.num_hiddens-self.cov2KerSize+1, 1)  
        edge_feat=edge_feat.permute(0,2,1)
        w5, b5 = vars[14], vars[15] 
        edge_feat=F.gelu(F.conv1d(edge_feat,w5,b5))
        edge_feat=F.adaptive_avg_pool1d(edge_feat, 512)
        edge_feat=edge_feat.contiguous().view(-1, 512)        
        w8, b8 = vars[16], vars[17]
        rel_pre = F.linear(edge_feat, w8, b8)  #layer 10 linear
        return rel_pre
    def zero_grad(self, vars=None): 
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars       
class Mymodel_GAT_cov_nobn(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens, num_edg_hiddens):
        super(Mymodel_GAT_cov_nobn, self).__init__()
        
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.num_edg_hiddens = num_edg_hiddens
        self.cov2KerSize = 50
        self.heads = 4
        
        
        ## layer0, gat_my1, var[0, 1, 2, 3]  w0,b0,attr_src, att_dst
        self.gat_my1 = GATConv_my(self.num_inputs, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w0 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_inputs]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w0)
        self.vars.append(w0) #w
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) #b
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        
        #layer 4 gcn_my2  var[4,5,6,7]   w2,b2,src,dst
        self.gat_my2 = GATConv_my(self.num_hiddens, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w2 = nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_hiddens]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) 
        self.vars.append(torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append(torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        
        #******* edge feature ####
        #layer 7 cov2 var[8,9]
        w4 = nn.Parameter(torch.ones([1,1,2,self.cov2KerSize]))
        torch.nn.init.kaiming_normal_(w4)
        self.vars.append(w4)
        self.vars.append(nn.Parameter(torch.zeros(1)))   
        
        #layer10 cov1 var[10,11]
        w6 = nn.Parameter(torch.ones([1,1,self.cov2KerSize]))
        torch.nn.init.kaiming_normal_(w6)
        self.vars.append(w6)
        self.vars.append(nn.Parameter(torch.zeros(1)))             
    
    
        #layer 12 linear var[12,13]
        w8 = nn.Parameter(torch.ones([self.num_outputs, 512]))
        torch.nn.init.kaiming_normal_(w8)
        self.vars.append(w8)
        self.vars.append(nn.Parameter(torch.zeros(self.num_outputs)))
        
        
    def forward(self, x, edge_index, vars = None, bn_training = True):
        
        if vars is None:
            vars = self.vars       
            
        w0, b0, att_src, att_dst = vars[0], vars[1], vars[2], vars[3]
#         print("edge_index.shape:", edge_index.shape)
        x = self.gat_my1(x, edge_index, w0, b0, att_src, att_dst)   # layer 0 gcn_my1
        x1 = F.relu(x)    #layer2  bn
        
        
        w2, b2, att_src1, att_dst1 =vars[4], vars[5],vars[6], vars[7]
        x = self.gat_my2(x1, edge_index, w2, b2, att_src1, att_dst1)  #layer 4 gcn_my2
        x = F.relu(x)     #layer6 gelu
        
        #### Get edge features
        edge_src = x[edge_index[0]]
        edge_dst = x[edge_index[1]]
        edge_feat = torch.cat((edge_src,edge_dst),axis = 1)
         
        w4, b4 = vars[8], vars[9]     
        edge_feat=edge_feat.view(-1,1,2,self.num_hiddens)
        edge_feat=F.relu(F.conv2d(edge_feat, w4, b4))
        edge_feat=edge_feat.view(-1,self.num_hiddens-self.cov2KerSize+1, 1)  
        edge_feat=edge_feat.permute(0,2,1)
        w5, b5 = vars[10], vars[11] 
        edge_feat=F.relu(F.conv1d(edge_feat,w5,b5))
        edge_feat=F.adaptive_avg_pool1d(edge_feat, 512)
        edge_feat=edge_feat.contiguous().view(-1, 512)        
        w8, b8 = vars[12], vars[13]
        rel_pre = F.linear(edge_feat, w8, b8)  #layer 10 linear
        return rel_pre
    def zero_grad(self, vars=None): 
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars     

class Mymodel_GAT_set2(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens, num_edg_hiddens):
        super(Mymodel_GAT_set2, self).__init__()
        
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.num_edg_hiddens = num_edg_hiddens
        self.heads = 1
        ## layer0, gat_my1, var[0, 1, 2, 3]  w0,b0,attr_src, att_dst
        self.gat_my1 = GATConv_my(self.num_inputs, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w0 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_inputs]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w0)
        self.vars.append(w0) #w
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) #b
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        #layer1, bn, var[4,5] , vars_bn[0,1]   w1,b1
        w1 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        #layer2 relu 
        #layer3 dropout
        
        
        #layer 4 gcn_my2  var[6,7,8,9]   w2,b2,src,dst
        self.gat_my2 = GATConv_my(self.num_hiddens, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w2 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_hiddens]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) 
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        
        #layer 5 bn    nvar[10,11]    vars_bn[2,3]
        w3 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w3)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        #layer 6 gelu
        
        
        #******* edge feature ####
        
        #layer 7 linear var[12,13]
        w4 = nn.Parameter(torch.ones([self.num_edg_hiddens,  self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w4)
        self.vars.append(w4)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))             
        # layer8 bn       var[14,15]  vars_bn[4,5]
        w5 = nn.Parameter(torch.ones(self.num_edg_hiddens))    # [ch_out]      
        self.vars.append(w5)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_edg_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_edg_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var])       
        
        #layer 9 relu
        #layer10 linear var[16,17]
        w6 = nn.Parameter(torch.ones([self.num_edg_hiddens,  self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w6)
        self.vars.append(w6)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))   
        
        
        # layer11 bn       var[18,19]  vars_bn[6,7]
        w7 = nn.Parameter(torch.ones(self.num_edg_hiddens))    # [ch_out]      
        self.vars.append(w7)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_edg_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_edg_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var])              
                          
        #layer 12 linear var[20,21]
        w8 = nn.Parameter(torch.ones([self.num_outputs, self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w8)
        self.vars.append(w8)
        self.vars.append(nn.Parameter(torch.zeros(self.num_outputs)))
        
        
    def forward(self, x, edge_index, vars = None, bn_training = True):
        
        if vars is None:
            vars = self.vars       
            
        w0, b0, att_src, att_dst = vars[0], vars[1], vars[2], vars[3]
#         print("edge_index.shape:", edge_index.shape)
        x = self.gat_my1(x, edge_index, w0, b0, att_src, att_dst)   # layer 0 gcn_my1
        
        w1, b1 = vars[4], vars[5]                
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]  
        x = F.batch_norm(x, running_mean, running_var, weight=w1, bias=b1, training=bn_training)   #layer 1 bn
#         print("x0.size:", x.size())
#         print("x.size:", x.size())
#         x1 = F.gelu(x)    #layer2  bn
        x1 = F.relu(x)    #layer2  bn
        
        
        w2, b2, att_src1, att_dst1 =vars[6], vars[7],vars[8], vars[9]
        x = self.gat_my2(x1, edge_index, w2, b2, att_src1, att_dst1)  #layer 4 gcn_my2
        w3, b3 = vars[10], vars[11]
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=w3, bias=b3, training=bn_training)   #layer5 bn
#         x = F.gelu(x + x1)     #layer6 gelu
        x = F.relu(x + x1)     #layer6 gelu
        
        #### Get edge features
        edge_src = x[edge_index[0]]
        edge_dst = x[edge_index[1]]
#         print("edge_src:",edge_src)
#         print("edge_dst:",edge_dst)

        edge_feat = torch.cat((edge_src,edge_dst),axis = 1)
         
        w4, b4 = vars[12], vars[13]     
        edge_feat = F.linear(edge_feat, w4, b4)   # layer 7 linear bn relu
        w5, b5 = vars[14], vars[15]
        running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
        edge_feat = F.batch_norm(edge_feat, running_mean, running_var, weight=w5, bias=b5, training=bn_training)  # layer 8 bn
#         edge_feat = F.gelu(edge_feat)       #layer9 relu
        edge_feat = F.relu(edge_feat)       #layer9 relu

        w6, b6 = vars[16], vars[17]     
        edge_feat = F.linear(edge_feat, w6, b6)   # layer 7 linear bn relu
        w7, b7 = vars[18], vars[19]
        running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
        edge_feat = F.batch_norm(edge_feat, running_mean, running_var, weight=w7, bias=b7, training=bn_training)  # layer 8 bn
#         edge_feat = F.gelu(edge_feat)       #layer9 relu
        edge_feat = F.relu(edge_feat)       #layer9 relu
        
        w8, b8 = vars[20], vars[21]
        rel_pre = F.linear(edge_feat, w8, b8)  #layer 10 linear
        return rel_pre
    def zero_grad(self, vars=None): 
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars 


class Mymodel_GAT_residue(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens, num_edg_hiddens):
        super(Mymodel_GAT_residue, self).__init__()
        
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.num_edg_hiddens = num_edg_hiddens
        self.heads = 4
        ## layer0, gat_my1, var[0, 1, 2, 3]  w0,b0,attr_src, att_dst
        self.gat_my1 = GATConv_my(self.num_inputs, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w0 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_inputs]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w0)
        self.vars.append(w0) #w
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) #b
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        #layer1, bn, var[4,5] , vars_bn[0,1]   w1,b1
        w1 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        #layer2 relu 
        #layer3 dropout
        
        
        #layer 4 gcn_my2  var[6,7,8,9]   w2,b2,src,dst
        self.gat_my2 = GATConv_my(self.num_hiddens, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w2 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_hiddens]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) 
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        
        #layer 5 bn    nvar[10,11]    vars_bn[2,3]
        w3 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w3)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        #layer 6 gelu
        
        
        #******* edge feature ####
        
        #layer 7 linear var[12,13]
        w4 = nn.Parameter(torch.ones([self.num_edg_hiddens,  self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w4)
        self.vars.append(w4)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))             
        # layer8 bn       var[14,15]  vars_bn[4,5]
        w5 = nn.Parameter(torch.ones(self.num_edg_hiddens))    # [ch_out]      
        self.vars.append(w5)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_edg_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_edg_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var])       
        
        #layer 9 relu
        #layer10 linear var[16,17]
        w6 = nn.Parameter(torch.ones([self.num_edg_hiddens,  self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w6)
        self.vars.append(w6)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))   
        
        
        # layer11 bn       var[18,19]  vars_bn[6,7]
        w7 = nn.Parameter(torch.ones(self.num_edg_hiddens))    # [ch_out]      
        self.vars.append(w7)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_edg_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_edg_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var])              
                          
        #layer 12 linear var[20,21]
        w8 = nn.Parameter(torch.ones([self.num_outputs, self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w8)
        self.vars.append(w8)
        self.vars.append(nn.Parameter(torch.zeros(self.num_outputs)))
        
        
    def forward(self, x, edge_index, vars = None, bn_training = True):
        
        if vars is None:
            vars = self.vars       
            
        w0, b0, att_src, att_dst = vars[0], vars[1], vars[2], vars[3]
#         print("edge_index.shape:", edge_index.shape)
        x = self.gat_my1(x, edge_index, w0, b0, att_src, att_dst)   # layer 0 gcn_my1
        
        w1, b1 = vars[4], vars[5]                
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]  
        x = F.batch_norm(x, running_mean, running_var, weight=w1, bias=b1, training=bn_training)   #layer 1 bn
#         print("x0.size:", x.size())
#         print("x.size:", x.size())
#         x1 = F.gelu(x)    #layer2  bn
        x1 = F.relu(x)    #layer2  bn
        
        
        w2, b2, att_src1, att_dst1 =vars[6], vars[7],vars[8], vars[9]
        x = self.gat_my2(x1, edge_index, w2, b2, att_src1, att_dst1)  #layer 4 gcn_my2
        w3, b3 = vars[10], vars[11]
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=w3, bias=b3, training=bn_training)   #layer5 bn
#         x = F.gelu(x + x1)     #layer6 gelu
        x = F.relu(x + x1)     #layer6 gelu
        
        #### Get edge features
        edge_src = x[edge_index[0]]
        edge_dst = x[edge_index[1]]
#         print("edge_src:",edge_src)
#         print("edge_dst:",edge_dst)

        edge_feat1 = torch.cat((edge_src,edge_dst),axis = 1)
         
        w4, b4 = vars[12], vars[13]     
        edge_feat = F.linear(edge_feat1, w4, b4)   # layer 7 linear bn relu
        w5, b5 = vars[14], vars[15]
        running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
        edge_feat = F.batch_norm(edge_feat, running_mean, running_var, weight=w5, bias=b5, training=bn_training)  # layer 8 bn
#         edge_feat = F.gelu(edge_feat)       #layer9 relu
        edge_feat1 = F.relu(edge_feat + edge_feat1)       #layer9 relu

        w6, b6 = vars[16], vars[17]     
        edge_feat = F.linear(edge_feat1, w6, b6)   # layer 7 linear bn relu
        w7, b7 = vars[18], vars[19]
        running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
        edge_feat = F.batch_norm(edge_feat, running_mean, running_var, weight=w7, bias=b7, training=bn_training)  # layer 8 bn
#         edge_feat = F.gelu(edge_feat)       #layer9 relu
        edge_feat = F.relu(edge_feat + edge_feat1)       #layer9 relu
        
        w8, b8 = vars[20], vars[21]
        rel_pre = F.linear(edge_feat, w8, b8)  #layer 10 linear
        return rel_pre
    def zero_grad(self, vars=None): 
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars 
















class Mymodel_GAT_gelu(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens, num_edg_hiddens):
        super(Mymodel_GAT_gelu, self).__init__()
        
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.num_edg_hiddens = num_edg_hiddens
        self.heads = 4
        ## layer0, gat_my1, var[0, 1, 2, 3]  w0,b0,attr_src, att_dst
        self.gat_my1 = GATConv_my(self.num_inputs, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w0 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_inputs]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w0)
        self.vars.append(w0) #w
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) #b
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        #layer1, bn, var[4,5] , vars_bn[0,1]   w1,b1
        w1 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        #layer2 relu 
        #layer3 dropout
        
        
        #layer 4 gcn_my2  var[6,7,8,9]   w2,b2,src,dst
        self.gat_my2 = GATConv_my(self.num_hiddens, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w2 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_hiddens]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) 
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        
        #layer 5 bn    nvar[10,11]    vars_bn[2,3]
        w3 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w3)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        #layer 6 gelu
        
        
        #******* edge feature ####
        
        #layer 7 linear var[12,13]
        w4 = nn.Parameter(torch.ones([self.num_edg_hiddens,  self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w4)
        self.vars.append(w4)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))             
        # layer8 bn       var[14,15]  vars_bn[4,5]
        w5 = nn.Parameter(torch.ones(self.num_edg_hiddens))    # [ch_out]      
        self.vars.append(w5)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_edg_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_edg_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var])       
        
        #layer 9 relu
        #layer10 linear var[16,17]
        w6 = nn.Parameter(torch.ones([self.num_edg_hiddens,  self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w6)
        self.vars.append(w6)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))   
        
        
        # layer11 bn       var[18,19]  vars_bn[6,7]
        w7 = nn.Parameter(torch.ones(self.num_edg_hiddens))    # [ch_out]      
        self.vars.append(w7)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_edg_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_edg_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var])              
                          
        #layer 12 linear var[20,21]
        w8 = nn.Parameter(torch.ones([self.num_outputs, self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w8)
        self.vars.append(w8)
        self.vars.append(nn.Parameter(torch.zeros(self.num_outputs)))
        
        
    def forward(self, x, edge_index, vars = None, bn_training = True):
        
        if vars is None:
            vars = self.vars       
            
        w0, b0, att_src, att_dst = vars[0], vars[1], vars[2], vars[3]
#         print("edge_index.shape:", edge_index.shape)
        x = self.gat_my1(x, edge_index, w0, b0, att_src, att_dst)   # layer 0 gcn_my1
        
        w1, b1 = vars[4], vars[5]                
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]  
        x = F.batch_norm(x, running_mean, running_var, weight=w1, bias=b1, training=bn_training)   #layer 1 bn
#         print("x0.size:", x.size())
#         print("x.size:", x.size())
        x1 = F.gelu(x)    #layer2  bn
        #x1 = F.relu(x)    #layer2  bn
        
        
        w2, b2, att_src1, att_dst1 =vars[6], vars[7],vars[8], vars[9]
        x = self.gat_my2(x1, edge_index, w2, b2, att_src1, att_dst1)  #layer 4 gcn_my2
        w3, b3 = vars[10], vars[11]
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=w3, bias=b3, training=bn_training)   #layer5 bn
        x = F.gelu(x + x1)     #layer6 gelu
        #x = F.relu(x + x1)     #layer6 gelu
        
        #### Get edge features
        edge_src = x[edge_index[0]]
        edge_dst = x[edge_index[1]]
#         print("edge_src:",edge_src)
#         print("edge_dst:",edge_dst)

        edge_feat = torch.cat((edge_src,edge_dst),axis = 1)
         
        w4, b4 = vars[12], vars[13]     
        edge_feat = F.linear(edge_feat, w4, b4)   # layer 7 linear bn relu
        w5, b5 = vars[14], vars[15]
        running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
        edge_feat = F.batch_norm(edge_feat, running_mean, running_var, weight=w5, bias=b5, training=bn_training)  # layer 8 bn
        edge_feat = F.gelu(edge_feat)       #layer9 relu
        #edge_feat = F.relu(edge_feat)       #layer9 relu

        w6, b6 = vars[16], vars[17]     
        edge_feat = F.linear(edge_feat, w6, b6)   # layer 7 linear bn relu
        w7, b7 = vars[18], vars[19]
        running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
        edge_feat = F.batch_norm(edge_feat, running_mean, running_var, weight=w7, bias=b7, training=bn_training)  # layer 8 bn
        edge_feat = F.gelu(edge_feat)       #layer9 relu
        #edge_feat = F.relu(edge_feat)       #layer9 relu
        
        w8, b8 = vars[20], vars[21]
        rel_pre = F.linear(edge_feat, w8, b8)  #layer 10 linear
        return rel_pre
    def zero_grad(self, vars=None): 
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars        




class Mymodel_GAT_noresidue(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens, num_edg_hiddens):
        super(Mymodel_GAT_noresidue, self).__init__()
        
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.num_edg_hiddens = num_edg_hiddens
        self.heads = 4
        ## layer0, gat_my1, var[0, 1, 2, 3]  w0,b0,attr_src, att_dst
        self.gat_my1 = GATConv_my(self.num_inputs, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w0 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_inputs]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w0)
        self.vars.append(w0) #w
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) #b
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        #layer1, bn, var[4,5] , vars_bn[0,1]   w1,b1
        w1 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        #layer2 relu 
        #layer3 dropout
        
        
        #layer 4 gcn_my2  var[6,7,8,9]   w2,b2,src,dst
        self.gat_my2 = GATConv_my(self.num_hiddens, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w2 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_hiddens]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) 
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        
        #layer 5 bn    nvar[10,11]    vars_bn[2,3]
        w3 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w3)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        #layer 6 gelu
        
        
        #******* edge feature ####
        
        #layer 7 linear var[12,13]
        w4 = nn.Parameter(torch.ones([self.num_edg_hiddens,  self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w4)
        self.vars.append(w4)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))             
        # layer8 bn       var[14,15]  vars_bn[4,5]
        w5 = nn.Parameter(torch.ones(self.num_edg_hiddens))    # [ch_out]      
        self.vars.append(w5)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_edg_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_edg_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var])       
        
        #layer 9 relu
        #layer10 linear var[16,17]
        w6 = nn.Parameter(torch.ones([self.num_edg_hiddens,  self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w6)
        self.vars.append(w6)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))   
        
        
        # layer11 bn       var[18,19]  vars_bn[6,7]
        w7 = nn.Parameter(torch.ones(self.num_edg_hiddens))    # [ch_out]      
        self.vars.append(w7)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_edg_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_edg_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var])              
                          
        #layer 12 linear var[20,21]
        w8 = nn.Parameter(torch.ones([self.num_outputs, self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w8)
        self.vars.append(w8)
        self.vars.append(nn.Parameter(torch.zeros(self.num_outputs)))
        
        
    def forward(self, x, edge_index, vars = None, bn_training = True):
        
        if vars is None:
            vars = self.vars       
            
        w0, b0, att_src, att_dst = vars[0], vars[1], vars[2], vars[3]
#         print("edge_index.shape:", edge_index.shape)
        x = self.gat_my1(x, edge_index, w0, b0, att_src, att_dst)   # layer 0 gcn_my1
        
        w1, b1 = vars[4], vars[5]                
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]  
        x = F.batch_norm(x, running_mean, running_var, weight=w1, bias=b1, training=bn_training)   #layer 1 bn
#         print("x0.size:", x.size())
#         print("x.size:", x.size())
#         x1 = F.gelu(x)    #layer2  bn
        x1 = F.relu(x)    #layer2  bn
        
        
        w2, b2, att_src1, att_dst1 =vars[6], vars[7],vars[8], vars[9]
        x = self.gat_my2(x1, edge_index, w2, b2, att_src1, att_dst1)  #layer 4 gcn_my2
        w3, b3 = vars[10], vars[11]
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=w3, bias=b3, training=bn_training)   #layer5 bn
        x = F.relu(x)     #layer6 gelu
        
        #### Get edge features
        edge_src = x[edge_index[0]]
        edge_dst = x[edge_index[1]]
#         print("edge_src:",edge_src)
#         print("edge_dst:",edge_dst)

        edge_feat = torch.cat((edge_src,edge_dst),axis = 1)
         
        w4, b4 = vars[12], vars[13]     
        edge_feat = F.linear(edge_feat, w4, b4)   # layer 7 linear bn relu
        w5, b5 = vars[14], vars[15]
        running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
        edge_feat = F.batch_norm(edge_feat, running_mean, running_var, weight=w5, bias=b5, training=bn_training)  # layer 8 bn
#         edge_feat = F.gelu(edge_feat)       #layer9 relu
        edge_feat = F.relu(edge_feat)       #layer9 relu

        w6, b6 = vars[16], vars[17]     
        edge_feat = F.linear(edge_feat, w6, b6)   # layer 7 linear bn relu
        w7, b7 = vars[18], vars[19]
        running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
        edge_feat = F.batch_norm(edge_feat, running_mean, running_var, weight=w7, bias=b7, training=bn_training)  # layer 8 bn
#         edge_feat = F.gelu(edge_feat)       #layer9 relu
        edge_feat = F.relu(edge_feat)       #layer9 relu
        
        w8, b8 = vars[20], vars[21]
        rel_pre = F.linear(edge_feat, w8, b8)  #layer 10 linear
        return rel_pre
    def zero_grad(self, vars=None): 
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars 

class Mymodel_GAT_drop(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens, num_edg_hiddens):
        super(Mymodel_GAT_drop, self).__init__()
        
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.num_edg_hiddens = num_edg_hiddens
        self.heads = 4
        ## layer0, gat_my1, var[0, 1, 2, 3]  w0,b0,attr_src, att_dst
        self.gat_my1 = GATConv_my(self.num_inputs, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w0 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_inputs]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w0)
        self.vars.append(w0) #w
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) #b
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        #layer1, bn, var[4,5] , vars_bn[0,1]   w1,b1
        w1 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        #layer2 relu 
        #layer3 dropout
        
        
        #layer 4 gcn_my2  var[6,7,8,9]   w2,b2,src,dst
        self.gat_my2 = GATConv_my(self.num_hiddens, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w2 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_hiddens]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) 
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        
        #layer 5 bn    nvar[10,11]    vars_bn[2,3]
        w3 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w3)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        #layer 6 gelu
        
        
        #******* edge feature ####
        
        #layer 7 linear var[12,13]
        w4 = nn.Parameter(torch.ones([self.num_edg_hiddens,  self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w4)
        self.vars.append(w4)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))             
        # layer8 bn       var[14,15]  vars_bn[4,5]
        w5 = nn.Parameter(torch.ones(self.num_edg_hiddens))    # [ch_out]      
        self.vars.append(w5)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_edg_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_edg_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var])       
        
        #layer 9 relu
        #layer10 linear var[16,17]
        w6 = nn.Parameter(torch.ones([self.num_edg_hiddens,  self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w6)
        self.vars.append(w6)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))   
        
        
        # layer11 bn       var[18,19]  vars_bn[6,7]
        w7 = nn.Parameter(torch.ones(self.num_edg_hiddens))    # [ch_out]      
        self.vars.append(w7)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_edg_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_edg_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var])              
                          
        #layer 12 linear var[20,21]
        w8 = nn.Parameter(torch.ones([self.num_outputs, self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w8)
        self.vars.append(w8)
        self.vars.append(nn.Parameter(torch.zeros(self.num_outputs)))
        
        
    def forward(self, x, edge_index, vars = None, bn_training = True, dropout_rate=0.2):
        
        if vars is None:
            vars = self.vars       
            
        w0, b0, att_src, att_dst = vars[0], vars[1], vars[2], vars[3]
#         print("edge_index.shape:", edge_index.shape)
        x = self.gat_my1(x, edge_index, w0, b0, att_src, att_dst)   # layer 0 gcn_my1
        
        w1, b1 = vars[4], vars[5]                
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]  
        x = F.batch_norm(x, running_mean, running_var, weight=w1, bias=b1, training=bn_training)   #layer 1 bn
#         print("x0.size:", x.size())
#         print("x.size:", x.size())
#         x1 = F.gelu(x)    #layer2  bn
        x = F.dropout(x, dropout_rate)
        x1 = F.relu(x)    #layer2  bn
        
        
        w2, b2, att_src1, att_dst1 =vars[6], vars[7],vars[8], vars[9]
        x = self.gat_my2(x1, edge_index, w2, b2, att_src1, att_dst1)  #layer 4 gcn_my2
        w3, b3 = vars[10], vars[11]
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=w3, bias=b3, training=bn_training)   #layer5 bn
#         x = F.gelu(x + x1)     #layer6 gelu
        x = F.dropout(x, dropout_rate)
        x = F.relu(x + x1)     #layer6 gelu
        
        #### Get edge features
        edge_src = x[edge_index[0]]
        edge_dst = x[edge_index[1]]
#         print("edge_src:",edge_src)
#         print("edge_dst:",edge_dst)

        edge_feat = torch.cat((edge_src,edge_dst),axis = 1)
         
        w4, b4 = vars[12], vars[13]     
        edge_feat = F.linear(edge_feat, w4, b4)   # layer 7 linear bn relu
        w5, b5 = vars[14], vars[15]
        running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
        edge_feat = F.batch_norm(edge_feat, running_mean, running_var, weight=w5, bias=b5, training=bn_training)  # layer 8 bn
#         edge_feat = F.gelu(edge_feat)       #layer9 relu
        edge_feat = F.dropout(edge_feat, dropout_rate)
        edge_feat = F.relu(edge_feat)       #layer9 relu

        w6, b6 = vars[16], vars[17]     
        edge_feat = F.linear(edge_feat, w6, b6)   # layer 7 linear bn relu
        w7, b7 = vars[18], vars[19]
        running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
        edge_feat = F.batch_norm(edge_feat, running_mean, running_var, weight=w7, bias=b7, training=bn_training)  # layer 8 bn
#         edge_feat = F.gelu(edge_feat)       #layer9 relu
        edge_feat = F.dropout(edge_feat, dropout_rate)
        edge_feat = F.relu(edge_feat)       #layer9 relu
        
        w8, b8 = vars[20], vars[21]
        rel_pre = F.linear(edge_feat, w8, b8)  #layer 10 linear
        return rel_pre
    def zero_grad(self, vars=None): 
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars 

class Mymodel_GAT_shrink(nn.Module):
    def __init__(self,num_inputs, num_outputs, num_hiddens, num_edg_hiddens):
        super(Mymodel_GAT_shrink, self).__init__()
        
        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.num_edg_hiddens = num_edg_hiddens
        self.heads = 4
        ## layer0, gat_my1, var[0, 1, 2, 3]  w0,b0,attr_src, att_dst
        self.gat_my1 = GATConv_my(self.num_inputs, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w0 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_inputs]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w0)
        self.vars.append(w0) #w
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) #b
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        #layer1, bn, var[4,5] , vars_bn[0,1]   w1,b1
        w1 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        #layer2 relu 
        #layer3 dropout
        
        
        #layer 4 gcn_my2  var[6,7,8,9]   w2,b2,src,dst
        self.gat_my2 = GATConv_my(self.num_hiddens, self.num_hiddens, self.heads)        # (in_feat, out_feat)
        w2 =  nn.Parameter(torch.ones([self.num_hiddens * self.heads, self.num_hiddens]))  #[out_feat, in_feat]
        torch.nn.init.kaiming_normal_(w2)
        self.vars.append(w2)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens))) 
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_src
        self.vars.append( torch.nn.init.kaiming_normal_(nn.Parameter(torch.Tensor(1, self.heads, self.num_hiddens)))) #att_dst
        
        
        
        #layer 5 bn    nvar[10,11]    vars_bn[2,3]
        w3 = nn.Parameter(torch.ones(self.num_hiddens))    # [ch_out]      
        self.vars.append(w3)
        self.vars.append(nn.Parameter(torch.zeros(self.num_hiddens)))        
        running_mean = nn.Parameter(torch.zeros(self.num_hiddens), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_hiddens), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var]) 
        
        #layer 6 gelu
        
        
        #******* edge feature ####
        
        #layer 7 linear var[12,13]
        w4 = nn.Parameter(torch.ones([self.num_edg_hiddens//2,  self.num_edg_hiddens]))
        torch.nn.init.kaiming_normal_(w4)
        self.vars.append(w4)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens//2)))             
        # layer8 bn       var[14,15]  vars_bn[4,5]
        w5 = nn.Parameter(torch.ones(self.num_edg_hiddens//2))    # [ch_out]      
        self.vars.append(w5)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens//2)))        
        running_mean = nn.Parameter(torch.zeros(self.num_edg_hiddens//2), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_edg_hiddens//2), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var])       
        
        #layer 9 relu
        #layer10 linear var[16,17]
        w6 = nn.Parameter(torch.ones([self.num_edg_hiddens//4,  self.num_edg_hiddens//2]))
        torch.nn.init.kaiming_normal_(w6)
        self.vars.append(w6)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens//4)))   
        # layer11 bn       var[18,19]  vars_bn[6,7]
        w7 = nn.Parameter(torch.ones(self.num_edg_hiddens//4))    # [ch_out]      
        self.vars.append(w7)
        self.vars.append(nn.Parameter(torch.zeros(self.num_edg_hiddens//4)))        
        running_mean = nn.Parameter(torch.zeros(self.num_edg_hiddens//4), requires_grad=False)    # must set requires_grad=False
        running_var = nn.Parameter (torch.ones(self.num_edg_hiddens//4), requires_grad=False) 
        self.vars_bn.extend([running_mean, running_var])              
                          
        #layer 12 linear var[20,21]
        w8 = nn.Parameter(torch.ones([self.num_outputs, self.num_edg_hiddens//4]))
        torch.nn.init.kaiming_normal_(w8)
        self.vars.append(w8)
        self.vars.append(nn.Parameter(torch.zeros(self.num_outputs)))
        
        
    def forward(self, x, edge_index, vars = None, bn_training = True):
        
        if vars is None:
            vars = self.vars       
            
        w0, b0, att_src, att_dst = vars[0], vars[1], vars[2], vars[3]
#         print("edge_index.shape:", edge_index.shape)
        x = self.gat_my1(x, edge_index, w0, b0, att_src, att_dst)   # layer 0 gcn_my1
        
        w1, b1 = vars[4], vars[5]                
        running_mean, running_var = self.vars_bn[0], self.vars_bn[1]  
        x = F.batch_norm(x, running_mean, running_var, weight=w1, bias=b1, training=bn_training)   #layer 1 bn
#         print("x0.size:", x.size())
#         print("x.size:", x.size())
#         x1 = F.gelu(x)    #layer2  bn
        x1 = F.relu(x)    #layer2  bn
        
        
        w2, b2, att_src1, att_dst1 =vars[6], vars[7],vars[8], vars[9]
        x = self.gat_my2(x1, edge_index, w2, b2, att_src1, att_dst1)  #layer 4 gcn_my2
        w3, b3 = vars[10], vars[11]
        running_mean, running_var = self.vars_bn[2], self.vars_bn[3]
        x = F.batch_norm(x, running_mean, running_var, weight=w3, bias=b3, training=bn_training)   #layer5 bn
#         x = F.gelu(x + x1)     #layer6 gelu
        x = F.relu(x + x1)     #layer6 gelu
        
        #### Get edge features
        edge_src = x[edge_index[0]]
        edge_dst = x[edge_index[1]]
#         print("edge_src:",edge_src)
#         print("edge_dst:",edge_dst)

        edge_feat = torch.cat((edge_src,edge_dst),axis = 1)
         
        w4, b4 = vars[12], vars[13]     
        edge_feat = F.linear(edge_feat, w4, b4)   # layer 7 linear bn relu
        w5, b5 = vars[14], vars[15]
        running_mean, running_var = self.vars_bn[4], self.vars_bn[5]
        edge_feat = F.batch_norm(edge_feat, running_mean, running_var, weight=w5, bias=b5, training=bn_training)  # layer 8 bn
#         edge_feat = F.gelu(edge_feat)       #layer9 relu
        edge_feat = F.relu(edge_feat)       #layer9 relu

        w6, b6 = vars[16], vars[17]     
        edge_feat = F.linear(edge_feat, w6, b6)   # layer 7 linear bn relu
        w7, b7 = vars[18], vars[19]
        running_mean, running_var = self.vars_bn[6], self.vars_bn[7]
        edge_feat = F.batch_norm(edge_feat, running_mean, running_var, weight=w7, bias=b7, training=bn_training)  # layer 8 bn
#         edge_feat = F.gelu(edge_feat)       #layer9 relu
        edge_feat = F.relu(edge_feat)       #layer9 relu
        
        w8, b8 = vars[20], vars[21]
        rel_pre = F.linear(edge_feat, w8, b8)  #layer 10 linear
        return rel_pre
    def zero_grad(self, vars=None): 
        """
        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars 


# edge_index = torch.tensor([[0, 1, 1, 2,2,3,1,3,4,2],
#                            [1, 0, 2, 1,3,2,3,1,2,4]], dtype=torch.long)
# # x = torch.rand((5,num_inputs), dtype=torch.float)
# num_inputs =6
# num_outputs = 8
# num_hiddens = 32
# num_edg_hiddens = 2 *num_hiddens


# feat_mtx = torch.Tensor([[1,2,3,4,5,3],[3,4,5,6,7,5],[7,4,5,6,7,6],[1,4,6,6,3,7],[3,4,5,6,7,0]])
# print("feat_mtx:", feat_mtx)
# net = Mymodel(num_inputs,num_outputs,num_hiddens, num_edg_hiddens)
# feat = net(feat_mtx, edge_index)
