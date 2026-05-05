import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
# from torch.utils.loader import DataLoader

# from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, GATv2Conv, ChebConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import MNISTSuperpixels

import torch_geometric.transforms as T

# num_genes = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## define the GAT class
class GAT(torch.nn.Module):
    def __init__(self, 
                    method, 
                    parallel, 
                    l2, 
                    decoder, 
                    poolsize, 
                    poolrate,
                    edge_weights, 
                    edge_attributes, 
                    num_gene,
                    num_mirna, 
                    omic_mode, 
                    num_classes, 
                    dropout_rate):

        super(GAT, self).__init__()
        self.omic_mode = omic_mode
        self.method = method
        self.parallel = parallel
        self.decoder = decoder
        self.l2 = l2
        self.poolsize = poolsize
        self.poolrate = poolrate
        self.edge_weights = edge_weights
        self.edge_attributes = edge_attributes
        self.hid = 6
        self.head = 8
        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.raised_dimension = 8
        self.concate_layer = 64

        if self.omic_mode < 3:
            self.num_features = 1
        else:
            self.num_features = 2

        self.pre_conv_linear_gene = nn.Linear(self.num_features, self.raised_dimension)
        self.pre_conv_linear_mirna = nn.Linear(1, self.raised_dimension)

        if method == 'gatv2':
            if self.edge_attributes:
                self.conv1 = GATv2Conv(self.raised_dimension, self.hid, heads=self.head, edge_dim=2)
                self.conv2 = GATv2Conv(self.hid * self.head, self.hid, heads=self.head, edge_dim=2)
            elif self.edge_weights:
                self.conv1 = GATv2Conv(self.raised_dimension, self.hid, heads=self.head, edge_dim=1)
                self.conv2 = GATv2Conv(self.hid * self.head, self.hid, heads=self.head, edge_dim=1)
            else:
                self.conv1 = GATv2Conv(self.raised_dimension, self.hid, heads=self.head)
                self.conv2 = GATv2Conv(self.hid * self.head, self.hid, heads=self.head)

        elif method == 'gat':
            if self.edge_attributes:
                self.conv1 = GATConv(self.raised_dimension, self.hid, heads=self.head, edge_dim=2)
                self.conv2 = GATConv(self.hid * self.head, self.hid, heads=self.head, edge_dim=2)
            elif self.edge_weights:
                self.conv1 = GATConv(self.raised_dimension, self.hid, heads=self.head, edge_dim=1)
                self.conv2 = GATConv(self.hid * self.head, self.hid, heads=self.head, edge_dim=1)
            else:
                self.conv1 = GATConv(self.raised_dimension, self.hid, heads=self.head)
                self.conv2 = GATConv(self.hid * self.head, self.hid, heads=self.head)

        self.linear_input = math.floor((self.num_gene + self.num_mirna) / self.poolsize) * self.hid * self.head
        print(self.linear_input)

        self.linear1 = nn.Linear(self.linear_input, self.linear_input//4)
        self.linear2 = nn.Linear(self.linear_input//4, self.concate_layer)

        if self.decoder:
            if self.num_features == 1:
                ## Omic mode: Exp, mi, Exp+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene+self.num_mirna)
            elif self.num_features == 2:
                ## omic_mode: Exp+CNV, Exp+CNV+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene*self.num_features + self.num_mirna)


        if self.parallel:

            parallel_input = self.raised_dimension*(self.num_gene+self.num_mirna)

            self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//4)
            self.parallel_linear2 = nn.Linear(parallel_input//4, self.concate_layer)
            self.classifier = nn.Linear(self.concate_layer*2, num_classes)
        else:
            self.classifier = nn.Linear(self.concate_layer, num_classes)

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x
    
    ## create the batch index for each nodes in the batch
    def create_batch_index(self, batches):
        batch_index = []
        for i in range(batches):
            batch_index += [i]*(self.num_gene+self.num_mirna)
        return(torch.Tensor(batch_index).type(torch.int64))
        
    def forward(self, x, edge_index, edge_weight):
        batches = x.shape[0]
        num_node = x.shape[1]
        
        if self.num_mirna == 0 or self.num_features == 1:
            x = self.pre_conv_linear_gene(x)
            x = F.relu(x)
        else:
            ## the second matrix cnv_data has padding
            x_exp_mirna = x[:,:,0]
            x_cnv = x[:,:,1]

            ## separate mirna from the rest
            x_cnv = x_cnv[:,:-100]
            x_exp = x_exp_mirna[:,:-100]

            x_cnv = x_cnv.view(batches,-1,1)
            x_exp = x_exp.view(batches,-1,1)
            x_gene = torch.cat([x_exp,x_cnv],dim=1)
            x_gene = x_gene.view(-1,self.num_features)
            x_mirna = x_exp_mirna[:,-100:]

            x_mirna = torch.flatten(x_mirna)
            x_mirna = x_mirna.view(-1, 1)

            x_gene = self.pre_conv_linear_gene(x_gene)
            x_gene = F.relu(x_gene)

            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            x_mirna = F.relu(x_mirna)
            # print(x_mirna.shape)

            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)

            x = torch.cat([x_gene,x_mirna],dim=1)



        x_parallel = x
        x = x.view(-1, self.raised_dimension)
        x_parallel = x_parallel.view(batches,-1)

        if self.edge_weights:
            x = self.conv1(x, edge_index, edge_weight)

            ## use different activation function based on the models
            x = F.leaky_relu(x)
        else:
            # print('Passing through Conv1 layer without edge_weight.')
            x = self.conv1(x, edge_index)

            ## use different activation function based on the models
            x = F.leaky_relu(x)

        if self.edge_weights:
            x = self.conv2(x, edge_index, edge_weight)

            x = F.leaky_relu(x)
        else:
            x = self.conv2(x, edge_index) ## output shape: [batches * num_node, hid * head]

            x = F.leaky_relu(x)

        ## pooling on the graph to reduce nodes
        x = x.view(batches, num_node, -1) ## output shape: [batches, num_node, hid * head]
        x = self.graph_max_pool(x, self.poolsize)   ## if "gat", then output shape: [batches, floor(num_node / poolsize), hid * head]
                                                        ## if "gcn", then output shape: [batches, floor(num_node / poolsize), hid]

        x = x.view(-1, self.hid * self.head) ## output shape:[batches * floor(num_node / poolsize), hid * head]

        x = x.view(batches, -1) ## output size: [batches, floor(num_node / poolsize) * hid * head]
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)

        if self.decoder:
            x_reconstruct = x
            x_reconstruct = self.decoder_1(x_reconstruct)
            x_reconstruct = F.relu(x_reconstruct)

            x_reconstruct  = nn.Dropout(self.dropout_rate)(x_reconstruct)
            x_reconstruct = self.decoder_2(x_reconstruct)

        if self.parallel:
            ## the two layer shallow FC network
            x_parallel = self.parallel_linear1(x_parallel)
            x_parallel = F.relu(x_parallel)

            x_parallel = self.parallel_linear2(x_parallel)
            x_parallel = F.relu(x_parallel)

            x = torch.cat((x,x_parallel),1)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.classifier(x)

        if self.decoder:
            return x_reconstruct, F.log_softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)
    
    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization, class_weights=None):
        if self.decoder:
            if self.num_mirna == 0 or self.num_features == 1:
                x_target = x_target.view(x_target.size()[0], -1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target)
            else:
                x_target_exp_mirna = x_target[:,:,0]
                x_target_cnv = x_target[:,:,1]

                ## separate mirna from the rest
                x_target_cnv = x_target_cnv[:,:-100]
                x_target_exp = x_target_exp_mirna[:,:-100]
                x_target_mirna = x_target_exp_mirna[:,-100:]
                x_target_flatten = torch.cat([x_target_exp, x_target_cnv, x_target_mirna], dim=1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target_flatten)
        else:
            loss1 = 0

        loss2 = nn.CrossEntropyLoss(weight=class_weights)(y, y_target)
        loss = 1*loss1 + 1*loss2

        if self.l2:
            l2_loss = 0.0
            for param in self.parameters():
                data = param* param
                l2_loss += data.sum()

            loss += 0.2* l2_regularization* l2_loss
        return loss


class GCN(torch.nn.Module):
    def __init__(self, 
                    method, 
                    parallel, 
                    l2, 
                    decoder, 
                    poolsize, 
                    poolrate,
                    edge_weights, 
                    edge_attributes, 
                    num_gene,
                    num_mirna, 
                    omic_mode, 
                    num_classes, 
                    dropout_rate):

        super(GCN, self).__init__()
        self.omic_mode = omic_mode
        self.method = method
        self.parallel = parallel
        self.decoder = decoder
        self.l2 = l2
        self.poolsize = poolsize
        self.poolrate = poolrate
        self.edge_weights = edge_weights
        self.edge_attributes = edge_attributes
        self.hid = 6
        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.raised_dimension = 8
        self.concate_layer = 64

        if self.omic_mode < 3:
            self.num_features = 1
        else:
            self.num_features = 2

        self.pre_conv_linear_gene = nn.Linear(self.num_features, self.raised_dimension)
        self.pre_conv_linear_mirna = nn.Linear(1, self.raised_dimension)
    
        if method == 'gcn':
            self.conv1 = ChebConv(self.raised_dimension, self.hid, K=5)
            self.conv2 = ChebConv(self.hid, self.hid, K=5)

        if self.poolsize <= 1:
            if method == 'gcn':
                self.linear_input = (self.num_gene + self.num_mirna) * self.hid
        else:
            if method == 'gcn':
                self.linear_input = math.floor((self.num_gene + self.num_mirna) / self.poolsize) * self.hid

        self.linear1 = nn.Linear(self.linear_input, self.linear_input//4)
        self.linear2 = nn.Linear(self.linear_input//4, self.concate_layer)

        if self.decoder:
            if self.num_features == 1:
                ## Omic mode: Exp, mi, Exp+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene+self.num_mirna)
            elif self.num_features == 2:
                ## omic_mode: Exp+CNV, Exp+CNV+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene*self.num_features + self.num_mirna)


        if self.parallel:

            parallel_input = self.raised_dimension*(self.num_gene + self.num_mirna)

            self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//4)
            self.parallel_linear2 = nn.Linear(parallel_input//4, self.concate_layer)
            self.classifier = nn.Linear(self.concate_layer*2, num_classes)
        else:
            self.classifier = nn.Linear(self.concate_layer, num_classes)

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x
    
    ## create the batch index for each nodes in the batch
    def create_batch_index(self, batches):
        batch_index = []
        for i in range(batches):
            batch_index += [i]*(self.num_gene + self.num_mirna)
        return(torch.Tensor(batch_index).type(torch.int64))
        
    def forward(self, x, edge_index, edge_weight):
        batches = x.shape[0]
        num_node = x.shape[1]
        
        if self.num_mirna == 0 or self.num_features == 1:
            x = self.pre_conv_linear_gene(x)
            x = F.relu(x)
        else:
            ## the second matrix cnv_data has padding
            x_exp_mirna = x[:,:,0]
            x_cnv = x[:,:,1]

            ## separate mirna from the rest
            x_cnv = x_cnv[:,:-100]
            x_exp = x_exp_mirna[:,:-100]

            x_cnv = x_cnv.view(batches,-1,1)
            x_exp = x_exp.view(batches,-1,1)
            x_gene = torch.cat([x_exp,x_cnv],dim=1)
            x_gene = x_gene.view(-1,self.num_features)
            x_mirna = x_exp_mirna[:,-100:]
            x_mirna = torch.flatten(x_mirna)
            x_mirna = x_mirna.view(-1, 1)

            x_gene = self.pre_conv_linear_gene(x_gene)
            x_gene = F.relu(x_gene)

            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            x_mirna = F.relu(x_mirna)

            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)

            x = torch.cat([x_gene,x_mirna],dim=1)



        x_parallel = x
        x = x.view(-1, self.raised_dimension)
        x_parallel = x_parallel.view(batches,-1)

        if self.edge_weights:
            x = self.conv1(x, edge_index, edge_weight)

            x = F.relu(x)
        else:
            x = self.conv1(x, edge_index)

            x = F.relu(x)

        if self.edge_weights:
            x = self.conv2(x, edge_index, edge_weight)

            x = F.relu(x)
        else:
            x = self.conv2(x, edge_index) ## output shape: [batches * num_node, hid * head]

            x = F.relu(x)

        ## pooling on the graph to reduce nodes
        x = x.view(batches, num_node, -1) ## output shape: [batches, num_node, hid * head]
        x = self.graph_max_pool(x, self.poolsize)   ## if "gat", then output shape: [batches, floor(num_node / poolsize), hid * head]
                                                    ## if "gcn", then output shape: [batches, floor(num_node / poolsize), hid]

        if self.method == 'gcn':
            x = x.view(-1, self.hid) ## output shape:[batches * floor(num_node / poolsize), hid]

        x = x.view(batches, -1) ## output size: [batches, floor(num_node / poolsize) * hid * head]
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)

        if self.decoder:
            x_reconstruct = x
            x_reconstruct = self.decoder_1(x_reconstruct)
            x_reconstruct = F.relu(x_reconstruct)

            x_reconstruct  = nn.Dropout(0.2)(x_reconstruct)
            x_reconstruct = self.decoder_2(x_reconstruct)

        if self.parallel:
            ## the two layer shallow FC network
            x_parallel = self.parallel_linear1(x_parallel)
            x_parallel = F.relu(x_parallel)
            x_parallel = self.parallel_linear2(x_parallel)
            x_parallel = F.relu(x_parallel)

            x = torch.cat((x,x_parallel),1)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.classifier(x)

        if self.decoder:
            return x_reconstruct, F.log_softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)
    
    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization, class_weights=None):
        if self.decoder:
            if self.num_mirna == 0 or self.num_features == 1:
                x_target = x_target.view(x_target.size()[0], -1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target)
            else:
                x_target_exp_mirna = x_target[:,:,0]
                x_target_cnv = x_target[:,:,1]

                ## separate mirna from the rest
                x_target_cnv = x_target_cnv[:,:-100]
                x_target_exp = x_target_exp_mirna[:,:-100]
                x_target_mirna = x_target_exp_mirna[:,-100:]
                x_target_flatten = torch.cat([x_target_exp, x_target_cnv, x_target_mirna], dim=1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target_flatten)
        else:
            loss1 = 0

        # https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        # weight should be a 1D Tensor assigning weight to each of the classes
        loss2 = nn.CrossEntropyLoss(weight=class_weights)(y, y_target)
        loss = 1*loss1 + 1*loss2

        if self.l2:
            l2_loss = 0.0
            for param in self.parameters():
                data = param* param
                l2_loss += data.sum()

            loss += 0.2* l2_regularization* l2_loss
        return loss


class Baseline(torch.nn.Module): # TODO tweak hyperparams, coz this doesn't run for me, my GPU memory explodes (WHY?) or run on GPU cluster
    def __init__(self, 
                    method, 
                    parallel, 
                    l2, 
                    decoder, 
                    poolsize, 
                    poolrate,
                    edge_weights, 
                    edge_attributes, 
                    num_gene,
                    num_mirna, 
                    omic_mode, 
                    num_classes, 
                    dropout_rate):

        super(Baseline, self).__init__()
        self.omic_mode = omic_mode
        self.method = method
        self.parallel = parallel
        self.decoder = decoder
        self.l2 = l2
        self.poolsize = poolsize
        self.poolrate = poolrate
        self.edge_weights = edge_weights
        self.edge_attributes = edge_attributes
        self.hid = 6
        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.raised_dimension = 8
        self.concate_layer = 64

        if self.omic_mode < 3: # no CNV, so just gene expression or just mirna
            self.num_features = 1
        else: # we have CNV, so can use CNV + gene expression
            self.num_features = 2

        # https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        self.pre_conv_linear_gene = nn.Linear(self.num_features, self.raised_dimension)
        self.pre_conv_linear_mirna = nn.Linear(1, self.raised_dimension) #

        parallel_input = self.raised_dimension*(self.num_gene + self.num_mirna) # 8 * (100 * 100) = 8000 (?!)

        self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//2) # 8000, 4000
        self.parallel_linear2 = nn.Linear(parallel_input//2, parallel_input//4) # 4000, 2000
        self.parallel_linear3 = nn.Linear(parallel_input//4, self.concate_layer) # 2000, 64
        self.classifier = nn.Linear(self.concate_layer, num_classes) # 64, 4
    
    ## create the batch index for each nodes in the batch
    def create_batch_index(self, batches):
        batch_index = []
        for i in range(batches):
            batch_index += [i]*(self.num_gene + self.num_mirna)
        return(torch.Tensor(batch_index).type(torch.int64))
        
    def forward(self, x, edge_index, edge_weight):

        batches = x.shape[0]
        num_node = x.shape[1]
        
        if self.num_mirna == 0 or self.num_features == 1:
            x = self.pre_conv_linear_gene(x)
            x = F.relu(x)
        else:
            ## the second matrix cnv_data has padding
            x_exp_mirna = x[:,:,0]
            x_cnv = x[:,:,1]

            ## separate mirna from the rest
            x_cnv = x_cnv[:,:-100]
            x_exp = x_exp_mirna[:,:-100]

            x_cnv = x_cnv.view(batches,-1,1)
            x_exp = x_exp.view(batches,-1,1)
            x_gene = torch.cat([x_exp,x_cnv],dim=1)
            x_gene = x_gene.view(-1,self.num_features)
            x_mirna = x_exp_mirna[:,-100:]
            x_mirna = torch.flatten(x_mirna)
            x_mirna = x_mirna.view(-1, 1)

            x_gene = self.pre_conv_linear_gene(x_gene)
            x_gene = F.relu(x_gene)

            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            x_mirna = F.relu(x_mirna)

            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)

            x = torch.cat([x_gene,x_mirna],dim=1)



        x_parallel = x
        x_parallel = x_parallel.view(batches,-1)
        
        x_parallel = self.parallel_linear1(x_parallel) # 8000, 4000
        x_parallel = F.relu(x_parallel)
        x_parallel = self.parallel_linear2(x_parallel) # 4000, 2000
        x_parallel = F.relu(x_parallel)
        x_parallel = self.parallel_linear3(x_parallel) # 2000, 64
        x_parallel = F.relu(x_parallel)

        x_parallel = F.dropout(x_parallel, p=self.dropout_rate, training=self.training)
        x_parallel = self.classifier(x_parallel) # 64, 4
        return F.log_softmax(x_parallel, dim=1)
    
    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization, class_weights=None):
        loss2 = nn.CrossEntropyLoss(weight=class_weights)(y, y_target)
        loss = 1*loss2

        return loss
    

class Baseline2(torch.nn.Module):
    # skip the dimension-raise step
    def __init__(self,
                    method,
                    parallel,
                    l2,
                    decoder,
                    poolsize,
                    poolrate,
                    edge_weights,
                    edge_attributes,
                    num_gene,
                    num_mirna,
                    omic_mode,
                    num_classes,
                    dropout_rate,
                    hidden_dim=512):

        super(Baseline2, self).__init__()
        self.omic_mode = omic_mode
        self.method = method
        self.parallel = parallel
        self.decoder = decoder
        self.l2 = l2
        self.poolsize = poolsize
        self.poolrate = poolrate
        self.edge_weights = edge_weights
        self.edge_attributes = edge_attributes
        self.hid = 6
        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.raised_dimension = 8
        self.concate_layer = 64

        if self.omic_mode < 3: # no CNV
            self.num_features = 1
        else: # we have CNV, so can use CNV + gene expression
            self.num_features = 2

        # self.pre_conv_linear_gene = nn.Linear(self.num_features, self.raised_dimension)
        # self.pre_conv_linear_mirna = nn.Linear(1, self.raised_dimension)

        # parallel_input = self.raised_dimension*(self.num_gene + self.num_mirna)

        # self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//2)
        # self.parallel_linear2 = nn.Linear(parallel_input//2, parallel_input//4)
        # self.parallel_linear3 = nn.Linear(parallel_input//4, self.concate_layer)
        # self.classifier = nn.Linear(self.concate_layer, num_classes)

        if self.num_mirna == 0: # no mirna, just gene features
            flat_input = self.num_gene * self.num_features # 100 * 2 = 200
        elif self.num_features == 1: # no CNV, so
            flat_input = self.num_gene + self.num_mirna # 100 + 100 = 200
        else: # have both gene features and mirna, so need to flatten and concat [gene exp, cnv, mirna]
            flat_input = self.num_gene * self.num_features + self.num_mirna # 100 * 2 + 100 = 300

        self.flat_input = flat_input

        # try simpler FC layers
        self.fc1 = nn.Linear(flat_input, hidden_dim) # 300, 512
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2) # 512, 256
        self.fc3 = nn.Linear(hidden_dim // 2, 64) # 256, 64
        self.classifier = nn.Linear(64, num_classes) # 64, 4

    def forward(self, x, edge_index, edge_weight):
        batches = x.shape[0]
        num_node = x.shape[1]

        if self.num_mirna == 0 or self.num_features == 1:
            # no mirna padding to handle, just flatten everything
            x_flat = x.view(batches, -1)
        else:
            # omic_mode 4: separate gene features (mRNA + CNV) from mirna (no CNV padding)
            # x[:,:,0] holds exp + mirna, x[:,:,1] holds cnv + padding
            x_exp_mirna = x[:, :, 0]
            x_cnv = x[:, :, 1]

            # strip the mirna section (last 100 cols) from the cnv channel (it's padding)
            x_cnv = x_cnv[:, :-self.num_mirna]
            x_exp = x_exp_mirna[:, :-self.num_mirna]
            x_mirna = x_exp_mirna[:, -self.num_mirna:]

            # concat flat: [exp, cnv, mirna] per sample
            x_flat = torch.cat([x_exp, x_cnv, x_mirna], dim=1)

        # connect the layers with ReLU and dropout
        x = F.relu(self.fc1(x_flat))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization, class_weights=None):
        loss2 = nn.CrossEntropyLoss(weight=class_weights)(y, y_target)
        loss = 1 * loss2

        if self.l2:
            l2_loss = 0.0
            for param in self.parameters():
                data = param * param
                l2_loss += data.sum()
            loss += 0.2 * l2_regularization * l2_loss
        return loss