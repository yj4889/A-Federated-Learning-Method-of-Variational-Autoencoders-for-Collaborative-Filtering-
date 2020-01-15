import os
#import shutil
#import sys
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
#% matplotlib inline
import seaborn as sn
import copy
#sn.set()
#import pandas as pd
#import bottleneck as bn
#import copy
import torch
#import torch.utils
#import torch.utils.data
#from torch import nn
#import torch.nn.functional as F
import torch.nn.functional as F

from utils.options import args_parser
from data.preprocess import data_preprocessing, load_train_data, load_tr_te_data
from model.Nets import MultiVAE
from model.Update import Update
from model.Update import evaluate
from model.Update import NDCG_binary_at_k_batch
from model.Update import Recall_at_k_batch

class EDGE(object):
    def __init__(self, data, state_net = 0):
        self.data = data
        self.state_net = state_net
      
def FedAvg(edges, num_edges, idxs):
    num_data = 0
    for i in range(0, num_edges):
        if edges[i].state_net != 0:
            num_data += edges[i].data.shape[0]

    w_avg = edges[idxs].state_net
   
    for k in w_avg.keys():
        w_avg[k] = w_avg[k]* edges[idxs].data.shape[0] / num_data
        for i in range(0, num_edges):
            if edges[i].state_net != 0:
                w_avg[k] += edges[i].state_net[k] * edges[i].data.shape[0]/num_data
                
    return w_avg

class LocalUpdate(object):
    def __init__(self, args, edge):
        self.args = args
        self.edge = edge
  
    def train(self, net, update_count, local_epochs):
        net.train()
        # train and update
        count = 0
        N = self.edge.data.shape[0]
        idxlist = list(range(N))
        

        np.random.shuffle(idxlist)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0)
        loss = []
        for iter in range(5):
            for batch_idx, start_idx in enumerate(range(0, N, 100)):
                end_idx = min(start_idx + 100, N)
                data = self.edge.data[idxlist[start_idx:end_idx]]
                data = torch.FloatTensor(data.toarray()).to(self.args.device)
            
                if self.args.total_anneal_steps > 0:
                    anneal = min(self.args.anneal_cap, 
                                    1. * update_count / self.args.total_anneal_steps)
                else:
                    anneal = self.args.anneal_cap
                

                optimizer.zero_grad()
                
                logits, mu, logvar = net(data)
        
                #loss definition (neg_ELBO = loss)
                neg_ll = -torch.mean(torch.sum(F.log_softmax(logits, 1) * data, -1))
                KL = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

                neg_ELBO = neg_ll + anneal * KL
                loss.append(neg_ELBO)
                neg_ELBO.backward()
            
                optimizer.step()
                count+=100
                if count % 500 == 0 and count != 0:
                    update_count += 1

                
            #log_softmax_var = F.log_softmax(logits, dim = 1)
        
        return net.to('cpu').state_dict(), sum(loss)/len(loss) , update_count
        
if __name__ == '__main__':
    num_edges = 100
    edges = []
    edges2 = []
    edges3 = []
    frac = 0.1
    local_epochs = 5
    
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    args.n_epochs = 30

    print(torch.cuda.is_available())

    if args.device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
   
    
    #isNotProcessed == 1 for preprocessing
    if args.preprocess == 1 :
        #Data Preprocessing
        data_preprocessing(args.data_dir)

    pro_dir = os.path.join(args.data_dir, 'pro_sg')

    #getting user's unique id
    unique_sid = list()
    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip()) #line.strip() 양쪽 공백과 \n을 삭제해준다.

    n_items = len(unique_sid)
        
    train_data = load_train_data(os.path.join(pro_dir, 'train.csv'), n_items)

    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(pro_dir, 'validation_tr.csv'), os.path.join(pro_dir, 'validation_te.csv'), n_items)
    
    #number of data size
    N = train_data.shape[0]
    N_vad = vad_data_tr.shape[0]
    idxlist = range(N)

    
    # training batch size
    batches_per_epoch = int(np.ceil(float(N) / args.batch_size)) #np.ceil() 은 요소들을 올림 ######################################################################

    N_vad = vad_data_tr.shape[0]

    idxlist_vad = range(N_vad)
    
    #Model size
    p_dims = [200, 600, n_items]

    #build model
    net_glob = MultiVAE(p_dims).to(args.device)
    net_glob2 = MultiVAE(p_dims).to(args.device)
    net_glob3 = MultiVAE(p_dims).to(args.device)
    print(net_glob)

    #training
    ndcgs_list = []
    r20_list = []
    r50_list = []
    loss_list = []

    best_ndcg = -np.inf
    update_count = 0.0
    update_count2 = 0.0
    update_count3 = 0.0
    #Federated Learning for VAEs
    # copy weights
    w_glob = net_glob.state_dict()
    w_glob2 = net_glob2.state_dict()
    w_glob3 = net_glob3.state_dict()

    start_idx = 0
    count = 0
    for i in range(0,num_edges):
        num_dataset = np.random.randint(800, 1100)
        end_idx = min(start_idx + num_dataset, N)
        edge = EDGE(data = train_data[start_idx : end_idx])
        edge2 = EDGE(data = train_data[start_idx : end_idx], state_net=copy.deepcopy(net_glob.to('cpu').state_dict()))
        edge3 = EDGE(data = train_data[start_idx : end_idx])

        edges.append(edge)
        edges2.append(edge2)
        edges3.append(edge3)

        start_idx = end_idx
        count=edges[i].data.shape[0]+count
    #print("count",count)
    #print("start_idx",start_idx)

    remain_data = int((N - start_idx) / num_edges)
    #print("remain_data", remain_data)
    st_idx = start_idx
    #print(sparse.vstack((train_data[0], train_data[1])))
    for i in range(0, num_edges):
        ed_idx = st_idx + remain_data
        if i == num_edges-1:
            edges[i].data = sparse.vstack((edges[i].data, train_data[st_idx:]))
            edges2[i].data = sparse.vstack((edges2[i].data, train_data[st_idx:]))
            edges3[i].data = sparse.vstack((edges3[i].data, train_data[st_idx:]))
        else:
            edges[i].data = sparse.vstack((edges[i].data, train_data[st_idx:ed_idx]))
            edges2[i].data = sparse.vstack((edges2[i].data, train_data[st_idx:ed_idx]))
            edges3[i].data = sparse.vstack((edges3[i].data, train_data[st_idx:ed_idx]))
        st_idx = ed_idx
    y = []
    for i in range(0, num_edges):
        y.append(edges[i].data.shape[0])

    
    plt.bar(np.arange(100), y)
    plt.ylim(1000, 1400)
    #plt.xticks(index, label, fontsize=15)
    plt.savefig('./log/ksc_bar_graph.png')



    