import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import logging.handlers
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import transforms

# device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# hyper parameters()
num_epochs = 10
learning_rate = 0.001

word_embed_dim = 1024
lstm_hidden_dim = 300
ntn_output_dim = 300
fc1_dim = 300
fc2_dim = 30
fc3_dim = 1

batch_size = 100
neg_samples = 4

# ===========================================================================
#logging outputs
LOG_FILE = 'MV_LSTM_wikiQA_bert.log'

handler = logging.handlers.RotatingFileHandler(LOG_FILE,encoding='utf-8') 
fmat = '%(asctime)s - %(levelname)s - %(message)s'

formatter = logging.Formatter(fmat)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter

logger = logging.getLogger('outputs')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.INFO)
# ===========================================================================
# load word embedding and word index
# word embedding path and word index path
# ===========================================================================
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
# ============================================================================    

def string_to_vec(query):
    query = query.split()
    len_query = len(query)
    
    wordo = query[0]
    query_veco = np.zeros((1,1,1024))
    if wordo in word_index:
        indexo = word_index[wordo]
        word_embedo = bert_content[indexo].split()
        word_embedo.pop(0)
        word_embedo_list = list(map(float,word_embedo))
        query_veco = np.array([[word_embedo_list]])
    
    for i in range(1,len_query):
        query_veci = np.zeros((1,1,1024))
        wordi = query[i]
        if wordi in word_index:
            indexi = word_index[wordi]
            word_embedi = bert_content[indexi].split()
            word_embedi.pop(0)
            word_embedi_list = list(map(float,word_embedi))
            query_veci = np.array([[word_embedi_list]])
        query_veco = np.concatenate((query_veco,query_veci),axis = 1) 
    return query_veco

# load wikiQA dataset
wikiQA = pd.read_csv('../../dataset/wikiQA/wikiQA.csv',header = 0)
# ============================================================================

def k_max_pooling(x,dim,k):
    index = x.topk(k,dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim,index)

class MV_LSTM(nn.Module):
    def __init__(self):
        super(MV_LSTM,self).__init__()
        self.bilstm = nn.LSTM(input_size = word_embed_dim,hidden_size = lstm_hidden_dim,batch_first = True,bidirectional = True)
        self.hidden_dim = lstm_hidden_dim
#         self.hidden = self.init_hidden()
        self.bilinear = nn.Bilinear(in1_features = lstm_hidden_dim*2,in2_features = lstm_hidden_dim*2,out_features = ntn_output_dim)
        self.ntn_fc = nn.Linear(in_features = lstm_hidden_dim*2,out_features = ntn_output_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features = fc1_dim,out_features = fc2_dim)
        self.fc2 = nn.Linear(in_features = fc2_dim,out_features = fc3_dim)
    
    def forward(self,query,pos_doc,neg_docs,hidden):
        # bilstm
        bilstm_query,hidden_ = self.bilstm(query,hidden)
        bilstm_pos_doc,hidden_ = self.bilstm(pos_doc,hidden)
        bilstm_neg_docs = []
        for neg_doc in neg_docs:
            bilstm_neg_doc,hidden_ = self.bilstm(neg_doc,hidden)
            bilstm_neg_docs.append(bilstm_neg_doc)
        
        # NTN
        # query pos_doc
        ntn_tensor_pos = Variable(torch.zeros(bilstm_query.shape[2],bilstm_pos_doc.shape[1],ntn_output_dim)).to(device)
        for i in range(bilstm_query.shape[2]):
            for j in range(bilstm_pos_doc.shape[1]):
                u = bilstm_query[1][i].reshape(1,1,-1)
                v = bilstm_pos_doc[1][j].reshape(1,1,-1)
                bilinear_output = self.bilinear(u,v)
                cat_uv = torch.cat((u,v),1)
                fc_out = self.ntn_fc(cat_uv)
                ntn_tensor_pos[i][j] = self.relu((bilinear_output + fc_out))
        
        #query neg_docs
        ntn_tensor_negs = []
        for bilstm_neg_doc in bilstm_neg_docs:
            ntn_tensor_neg = Variable(torch.zeros(bilstm_query.shape[2],bilstm_neg_doc.shape[1],ntn_output_dim)).to(device) 
            for i in range(bilstm_query.shape[2]):
                for j in range(bilstm_neg_doc.shape[1]):
                    u = bilstm_query[1][i].reshape(1,1,-1)
                    v = bilstm_neg_doc[1][j].reshape(1,1,-1)
                    bilinear_output = self.bilinear(u,v)
                    cat_uv = torch.cat((u,v),1)
                    fc_out = self.ntn_fc(cat_uv)
                    ntn_tensor_neg[i][j] = self.relu((bilinear_output + fc_out))
            ntn_tensor_negs.append(ntn_tensor_neg)
        
        # k_max_pooling
        q_query_pos = k_max_pooling(ntn_tensor_pos,dim = 1,k = 1)
        q_query_pos = k_max_pooling(q_query_pos,dim = 1, k = 1)
        q_query_pos = q_query_pos
        
        q_query_negs = []
        for ntn_tensor_neg in ntn_tensor_negs:
            q_query_neg = k_max_pooling(ntn_tensor_neg,dim = 1,k = 1)
            q_query_neg = k_max_pooling(q_query_neg,dim = 1,k = 1)
            q_query_neg = q_query_neg
            q_query_negs.append(q_query_neg)
        
        # fully connection s
        fc1_output_qp = self.fc1(q_query_pos)
        fc1_output_qns = [self.fc1(q_query_neg) for q_query_neg in q_query_negs]
        
        s_query_pos = [self.fc2(fc1_output_qp)]
        s_query_negs = [self.fc2(fc1_output_qn) for fc1_output_qn in fc1_output_qns]
        s = s_query_pos + s_query_negs
        s = torch.stack(s)
        #print(s)
        return s

model = MV_LSTM().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(),lr = learning_rate)

# train the model
for epoch in range(num_epochs):
    total_step = wikiQA.shape[0] - 40
    for i in range(0,total_step,100):
        querys = []
        pos_docs = []
        neg_docs = []
        for k in range(i,100+i):
            query = wikiQA.loc[k,'query']
            pos = wikiQA.loc[k,'pos']
            neg1 = wikiQA.loc[k,'neg1']
            neg2 = wikiQA.loc[k,'neg2']
            neg3 = wikiQA.loc[k,'neg3']
            neg4 = wikiQA.loc[k,'neg4']
            
            # (1,n,1024)
            query_vec = string_to_vec(query)
            pos_vec = string_to_vec(pos)
            neg1_vec = string_to_vec(neg1)
            neg2_vec = string_to_vec(neg2)
            neg3_vec = string_to_vec(neg3)
            neg4_vec = string_to_vec(neg4)
            
            querys.append(query_vec)
            pos_docs.append(pos_vec)
            neg_docs.append([neg1_vec,neg2_vec,neg3_vec,neg4_vec])
        
        log_string = 'processing ' + str(k+1) + 'th sample,word embed (1,n,1024) dim...'
        logger.info(log_string)
        
        y = np.ndarray(1)
        y[0] = 0
        y = Variable(torch.from_numpy(y).long()).to(device)
        
        for m in range(batch_size):
            querys[m] = Variable(torch.from_numpy(querys[m]).float()).to(device)
            pos_docs[m] = Variable(torch.from_numpy(pos_docs[m]).float()).to(device)
            
            for v in range(neg_samples):
                neg_docs[m][v] = Variable(torch.from_numpy(neg_docs[m][v]).float()).to(device)
            hidden = model.init_hidden()
            
            # forward pass
            y_pred = model(querys[m],pos_docs[m],[neg_docs[m][v] for v in range(neg_samples)],hidden)
            loss = criterion(y_pred.reshape(1,neg_samples+1),y)
            
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print loss
        log_string = 'Epoch: '+ str(epoch+1) + '  Sample Start: ' + str(i) + ' Loss: ' + str(loss.item())
        logger.info(log_string)
        if loss.item() <= 0.001:
            string_info = 'loss <= 0.001,finished training...'
            logger.info(string_info)
            break
    if loss.item() <= 0.001:
        string_info = 'loss <= 0.001,finished training...'
        logger.info(string_info)
        break

# save thr model
torch.save(model.state_dict(), 'MV_LSTM_wikiQA_bert.ckpt')
