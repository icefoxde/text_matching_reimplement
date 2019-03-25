import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import logging.handlers
from torch.autograd import Variable

# device configuration
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

# hypter parameters
num_epochs = 20
batch_size = 100
neg_samples = 4
learning_rate = 0.001
conv2d_L1_feature_map = 8
conv2d_L1_kernel_size = 3
conv2d_L2_feature_map = 16
conv2d_L2_kernel_size = 2

dynamic_max_pooling_size = 8
max_pooling_size = 4
linear_L1_input_size = conv2d_L2_feature_map * 16
linear_L2_input_size = 100

# ===========================================================================
# logging output
LOG_FILE = 'match_pyramid_wikiQA_bert_train.log'

handler = logging.handlers.RotatingFileHandler(LOG_FILE,encoding='utf-8') 
fmat = '%(asctime)s - %(levelname)s - %(message)s'

formatter = logging.Formatter(fmat)                # 实例化formatter
handler.setFormatter(formatter)                   # 为handler添加formatter

logger = logging.getLogger('outputs')           # 获取名为tst的logger
logger.addHandler(handler)                     # 为logger添加handler
logger.setLevel(logging.INFO)
# ===========================================================================
# load wikiQA dataset
wikiQA = pd.read_csv('../../dataset/wikiQA/wikiQA.csv',header = 0)
# load en bert word embedding and word index
en_bert_path = '../../dataset/bert/en_bert_model/en_bert_wordembed.txt'
en_word_index_path = '../../dataset/bert/en_bert_model/en_bert_word_index.txt'
# load word embedding and word index
with open(en_bert_path,'r') as fc:
    bert_content = fc.readlines()

with open(en_word_index_path,'r') as fw:
    index_content = fw.readlines()
    
word_index = dict()
len_index_content = len(index_content)
for i in range(len_index_content):
    string = index_content[i].split()
    word_index[string[0]] = int(string[1])
# ===========================================================================
# string to vector
def string_to_vec(query):
    query = query.split()
    len_query = len(query)
    if len_query < 8:
        query_vec = np.zeros((1,14,1024))
        for i in range(len_query):
            wordi = query[i]
            if wordi in word_index:
                indexi = word_index[wordi]
                word_embedi = bert_content[indexi].split()
                word_embedi.pop(0)
                word_embedi_list = list(map(float,word_embedi))
                query_vec[0][i] = np.array(word_embedi_list)
    
    else:
        query_vec = np.zeros((1,len_query,1024))
        for i in range(len_query):
            wordi = query[i]
            if wordi in word_index:
                indexi = word_index[wordi]
                word_embedi = bert_content[indexi].split()
                word_embedi.pop(0)
                word_embedi_list = list(map(float,word_embedi))
                query_vec[0][i] = np.array(word_embedi_list)
    
    return query_vec
# ===================================================
def k_max_pooling(x,dim,k):
    index = x.topk(k,dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim,index)

class match_pyramid(nn.Module):
    def __init__(self):
        super(match_pyramid,self).__init__()
        
        # convolution layer
        self.conv2d_L1 = nn.Conv2d(in_channels = 1,out_channels = conv2d_L1_feature_map,kernel_size = conv2d_L1_kernel_size,stride = 1)
        self.relu_conv_L1 = nn.ReLU()
        self.conv2d_L2 = nn.Conv2d(in_channels = conv2d_L1_feature_map,out_channels = conv2d_L2_feature_map,kernel_size = conv2d_L2_kernel_size,stride = 1)
        self.relu_conv_L2 = nn.ReLU()
        
        # fully connection layer
        self.linear_L1 = nn.Linear(in_features = linear_L1_input_size,out_features = linear_L2_input_size)
        self.relu_fc_L1 = nn.ReLU()
        self.linear_L2 = nn.Linear(in_features = linear_L2_input_size,out_features = 1)
        self.relu_fc_L2 = nn.ReLU()
    
    def conv_pooling(self,matrix_pos,num_layers,pool_size):
        if num_layers == 1:
            pos_L1_conv_output = self.relu_conv_L1(self.conv2d_L1(matrix_pos))
            pos_L1_pool_output = k_max_pooling(pos_L1_conv_output,2,pool_size)
            pos_cp_output = k_max_pooling(pos_L1_pool_output,3,pool_size)
            
        elif num_layers == 2:
            pos_L2_conv_output = self.relu_conv_L2(self.conv2d_L2(matrix_pos))
            pos_L2_pool_output = k_max_pooling(pos_L2_conv_output,2,pool_size)
            pos_cp_output = k_max_pooling(pos_L2_pool_output,3,pool_size)
            
        else:
            print('num_layers not in 1-2')
        
        return pos_cp_output
    
    def mlp(self,query):
        query_L1_linear_output = self.relu_fc_L1(self.linear_L1(query))
        query_L2_linear_output = self.relu_fc_L2(self.linear_L2(query_L1_linear_output))
        return query_L2_linear_output.squeeze(0)
    
    def forward(self,query,pos_doc,neg_docs):
        # matrix 
        matrix_pos = self.calculate_matrix(query,pos_doc)
        matrix_negs = [self.calculate_matrix(query,neg_doc) for neg_doc in neg_docs]
        print('matrix_pos: ',matrix_pos.shape)
        print('matrix_negs[0]: ',matrix_negs[0].shape)
        
        matrix_pos_reshape = matrix_pos.reshape(1,1,matrix_pos.shape[0],matrix_pos.shape[1])
        matrix_negs_reshape = [matrix_neg.reshape(1,1,matrix_neg.shape[0],matrix_neg.shape[1]) for matrix_neg in matrix_negs]
        
        # pos conv + pooling
        pos_L1_output = self.conv_pooling(matrix_pos_reshape,1,dynamic_max_pooling_size)
        pos_L2_output = self.conv_pooling(pos_L1_output,2,max_pooling_size)
        pos_output = pos_L2_output.reshape(1,-1)
        
        #negs conv + pooling
        negs_L1_output = [self.conv_pooling(matrix_neg_reshape,1,dynamic_max_pooling_size) for matrix_neg_reshape in matrix_negs_reshape]
        negs_L2_output = [self.conv_pooling(neg_L1_output,2,max_pooling_size) for neg_L1_output in negs_L1_output]
        negs_output = [neg_L2_output.reshape(1,-1) for neg_L2_output in negs_L2_output]
        
        # fully connection
        pos_mlp = [self.mlp(pos_output)]
        negs_mlp = [self.mlp(neg_output) for neg_output in negs_output]
        
        s = pos_mlp + negs_mlp
        s = torch.stack(s)
        
        return s
        
        
model = match_pyramid().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(),lr = learning_rate)

# ====================================================================
# wikiQA data to vector
querys = []
pos_docs = []
neg_docs = []
total_step = wikiQA.shape[0] - 40

for k in range(10):
    
    # origin data
    query = wikiQA.loc[k,'query']
    pos = wikiQA.loc[k,'pos']
    neg1 = wikiQA.loc[k,'neg1']
    neg2 = wikiQA.loc[k,'neg2']
    neg3 = wikiQA.loc[k,'neg3']
    neg4 = wikiQA.loc[k,'neg4']
    
    # vector 
    query_vec = string_to_vec(query)
    pos_vec = string_to_vec(pos)
    neg1_vec = string_to_vec(neg1)
    neg2_vec = string_to_vec(neg2)
    neg3_vec = string_to_vec(neg3)
    neg4_vec = string_to_vec(neg4)
    
    # Variable 
    query_vec = Variable(torch.from_numpy(query_vec).float()).to(device)
    pos_vec = Variable(torch.from_numpy(pos_vec).float()).to(device)
    neg1_vec = Variable(torch.from_numpy(neg1_vec).float()).to(device)
    neg2_vec = Variable(torch.from_numpy(neg2_vec).float()).to(device)
    neg3_vec = Variable(torch.from_numpy(neg3_vec).float()).to(device)
    neg4_vec = Variable(torch.from_numpy(neg4_vec).float()).to(device)
    
    querys.append(query_vec)
    pos_docs.append(pos_vec)
    neg_docs.append([neg1_vec,neg2_vec,neg3_vec,neg4_vec])

# ========================================================
for epoch in range(num_epochs):
    y = np.ndarray(1)
    y[0] = 0
    y = Variable(torch.from_numpy(y).long()).to(device)
    for j in range(0,10):

        # forward pass
        y_pred = model(querys[j],pos_docs[j],neg_docs[j])
        loss = criterion(y_pred.reshape(1,neg_samples+1),y)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print loss
        print('epoch: {}, steps: {}/{}, loss: {}'.format(epoch,j,total_step,loss.item()))

# # save thr model
# torch.save(model.state_dict(), 'match_pyramid_wikiQA_bert.ckpt')
