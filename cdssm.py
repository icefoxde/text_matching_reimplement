import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import logging.handlers
import torch.nn.functional as F
from torch.autograd import Variable
from l3wtransformer import L3wTransformer
from torchvision.transforms import transforms

# device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# hyper parameters
window_size =3
num_epochs = 50
batch_size = 10
filter_length = 1
sample_size = 1040
letter_ngram_size = 3
learning_rate = 0.0001

max_pooling_dim = 300         
latent_semantic_dim = 128     
select_negSample = 4

total_letter_ngram = int(1*1e4)
word_depth = window_size * total_letter_ngram


def k_max_pooling(x,dim,k):
    index = x.topk(k,dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim,index)

class CDSSM(nn.Module):
    def __init__(self):
        super(CDSSM,self).__init__()
        
        # laysers of query
        self.query_conv = nn.Conv1d(word_depth,max_pooling_dim,filter_length)
        self.query_fc = nn.Linear(max_pooling_dim,latent_semantic_dim)
        
        # layers of docs
        self.docs_conv = nn.Conv1d(word_depth,max_pooling_dim,filter_length)
        self.docs_fc = nn.Linear(max_pooling_dim,latent_semantic_dim)
        
        # learning gamma
        self.gamma = nn.Conv2d(1,2,1)
    
    def forward(self,query,pos_doc,neg_docs):
        # =================================================================
        query = query.transpose(2,1)
        query_conv_vec = torch.tanh(self.query_conv(query))
        query_max_pooling = k_max_pooling(query_conv_vec,2,1)
        query_max_pooling = query_max_pooling.transpose(1,2)
        query_semantic_vec = torch.tanh(self.query_fc(query_max_pooling))
        query_semantic_vec = query_semantic_vec.reshape(latent_semantic_dim)
        
        # ==================================================================
        pos_doc = pos_doc.transpose(2,1)
        pos_doc_conv_vec = torch.tanh(self.docs_conv(pos_doc))
        pos_doc_max_pooling = k_max_pooling(pos_doc_conv_vec,2,1)
        pos_doc_max_pooling = pos_doc_max_pooling.transpose(2,1)
        pos_doc_semantic_vec = torch.tanh(self.docs_fc(pos_doc_max_pooling))
        pos_doc_semantic_vec = pos_doc_semantic_vec.reshape(latent_semantic_dim)
        
        # =====================================================================
        neg_docs = [neg_doc.transpose(2,1) for neg_doc in neg_docs]
        neg_docs_conv_vec = [torch.tanh(self.docs_conv(neg_doc)) for neg_doc in neg_docs]
        neg_docs_max_pooling = [k_max_pooling(neg_doc_conv_vec,2,1) for neg_doc_conv_vec in neg_docs_conv_vec]
        neg_docs_max_pooling = [neg_doc_max_pooling.transpose(1,2) for neg_doc_max_pooling in neg_docs_max_pooling]
        neg_docs_semantic_vec = [torch.tanh(self.docs_fc(neg_doc_max_pooling)) for neg_doc_max_pooling in neg_docs_max_pooling]
        neg_docs_semantic_vec = [neg_doc_semantic_vec.reshape(latent_semantic_dim) for neg_doc_semantic_vec in neg_docs_semantic_vec]
        
        # ===========================================================================================================================
        dots = [query_semantic_vec.dot(pos_doc_semantic_vec)]
        dots = torch.stack(dots)
        dots = dots + [query_semantic_vec.dot(neg_doc_semantic_vec) for neg_doc_semantic_vec in neg_docs_semantic_vec]
        return dots

model = CDSSM().to(device)

#logging outputs
LOG_FILE = 'outputs.log'

handler = logging.handlers.RotatingFileHandler(LOG_FILE,encoding='utf-8')  # 实例化handler
fmat = '%(asctime)s - %(levelname)s - %(message)s'

formatter = logging.Formatter(fmat)  # 实例化formatter
handler.setFormatter(formatter)  # 为handler添加formatter

logger = logging.getLogger('outputs')  # 获取名为tst的logger
logger.addHandler(handler)  # 为logger添加handler
logger.setLevel(logging.INFO)

#==============================================================
wikiQA = pd.read_csv('../dataset/wikiQA/wikiQA.csv',header = 0)
#==============================================================
lt = L3wTransformer(max_ngrams = 3999)

sentences = []
for i in range(wikiQA.shape[0]):
    query = wikiQA.loc[i,'query']
    pos = wikiQA.loc[i,'pos']
    neg1 = wikiQA.loc[i,'neg1']
    neg2 = wikiQA.loc[i,'neg2']
    neg3 = wikiQA.loc[i,'neg3']
    neg4 = wikiQA.loc[i,'neg4']
    sentences.append(query)
    sentences.append(pos)
    sentences.append(neg1)
    sentences.append(neg2)
    sentences.append(neg3)
    sentences.append(neg4)

# build lookup table
lt.fit_on_texts(sentences)

def string_to_vec(query):
    query = query.split()
    qlen = len(query) -2
    query_n_word = []
    for i in range(qlen):
        word1 = query[i]
        word2 = query[i+1]
        word3 = query[i+2]
        
        i1 = lt.texts_to_sequences([word1])
        query_vec1 = np.zeros((1,1,10000))
        for index in i1[0]:
            if index < 10000:
                query_vec1[0][0][index] += 1
        
        i2 = lt.texts_to_sequences([word2])
        query_vec2 = np.zeros((1,1,10000))
        for index in i2[0]:
            if index < 10000:
                query_vec2[0][0][index] += 1
        
        i3 = lt.texts_to_sequences([word3])
        query_vec3 = np.zeros((1,1,10000))
        for index in i3[0]:
            if index < 10000:
                query_vec3[0][0][index] += 1
        
        query_vec = np.concatenate((query_vec1,query_vec2),axis = 2)
        query_vec = np.concatenate((query_vec,query_vec3),axis = 2)
        query_n_word.append(query_vec)

    query_n_vec = query_n_word[0]
    for i in range(1,qlen):
        query_n_vec = np.concatenate((query_n_vec,query_n_word[i]),axis = 1)
    return query_n_vec
#=============================================================================
# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)


# word hashing and train
for epoch in range(num_epochs):
    total_step = wikiQA.shape[0] # 1040
    for i in range(0,total_step,10):
        querys = []
        pos_docs = []
        neg_docs = []
        for k in range(0,10+i):
            query ='<s> ' +  wikiQA.loc[k,'query'] + ' <s>'
            pos ='<s> ' +  wikiQA.loc[k,'pos'] + ' <s>'
            neg1 ='<s> ' +  wikiQA.loc[k,'neg1'] + ' <s>'
            neg2 ='<s> ' +  wikiQA.loc[k,'neg2'] + ' <s>'
            neg3 ='<s> ' +  wikiQA.loc[k,'neg3'] + ' <s>'
            neg4 ='<s> ' +  wikiQA.loc[k,'neg4'] + ' <s>'

            query_n_vec = string_to_vec(query)
            pos_n_vec = string_to_vec(pos)
            neg1_n_vec = string_to_vec(neg1)
            neg2_n_vec = string_to_vec(neg2)
            neg3_n_vec = string_to_vec(neg3)
            neg4_n_vec = string_to_vec(neg4)
            querys.append(query_n_vec)
            pos_docs.append(pos_n_vec)
            neg_docs.append([neg1_n_vec,neg2_n_vec,neg3_n_vec,neg4_n_vec])    

            log_string = 'processing ' + str(k+1) + 'th sample,word hashing...'
            logger.info(log_string)
        
        # train the model
        y = np.ndarray(1)
        y[0] = 0
        y = Variable(torch.from_numpy(y).long()).to(device)
        for m in range(batch_size):
            
            querys[m] = Variable(torch.from_numpy(querys[m]).float()).to(device)
            pos_docs[m] = Variable(torch.from_numpy(pos_docs[m]).float()).to(device)
            for v in range(select_negSample):
                neg_docs[m][v] = Variable(torch.from_numpy(neg_docs[m][v]).float()).to(device)
            
            # forward pass
            y_pred = model(querys[m],pos_docs[m],[neg_docs[m][v] for v in range(select_negSample)])
            loss = criterion(y_pred.reshape(1,select_negSample+1),y)
            
            # backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            log_string = 'Epoch: '+ str(epoch+1) + '  Sample Start: ' + str(i) + 'Loss: ' + str(loss.item())
            logger.info(log_string)

# save thr model
torch.save(model.state_dict(), 'CDSSM model params.ckpt')