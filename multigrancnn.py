import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hypter parameters
num_epochs = 10
batch_size = 100
neg_samples = 4
learning_rate = 0.001
word_embed_dim = 1024

gpconv_L1_input_dim = 1024
gpconv_L1_output_dim = 300
gpconv_L2_input_dim = 300
gpconv_L2_output_dim = 300
gpconv_L3_input_dim = 300
gpconv_L3_output_dim = 300

bilin_input_dim = 300
bilin_output_dim = 1
mf_model_pool_dim = 40

mfcnn_L1_input_dim = 1
mfcnn_L1_output_dim = 50
mfcnn_L1_pool_HW = 40
mfcnn_L2_input_dim = 1
mfcnn_L2_output_dim = 50
mfcnn_L2_pool_HW = 40
mfcnn_L3_input_dim = 1
mfcnn_L3_output_dim = 50
mfcnn_L3_pool_HW = 40


# ========================================================
# load dataset
querys = []
pos_docs = []
neg_docs = []
for i in range(batch_size):
    query = torch.ones(1,10,1024)
    pos_doc = query + torch.rand(1,10,1024)
    neg_doc = torch.rand(1,14,1024)

    querys.append(query)
    pos_docs.append(pos_doc)
    neg_docs.append([neg_doc for i in range(neg_samples)])
# ========================================================
def k_max_pooling(x,dim,k):
    index = x.topk(k,dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim,index)

class multigrancnn(nn.Module):
    def __init__(self):
        super(multigrancnn,self).__init__()
        # gpCNN
        # siamese strcture
        # query conv
        self.query_conv_L1 = nn.Conv1d(in_channels = gpconv_L1_input_dim,out_channels = gpconv_L1_output_dim,kernel_size = 2)
        self.query_tanh_L1 = nn.Tanh()
        self.query_conv_L2 = nn.Conv1d(in_channels = gpconv_L2_input_dim,out_channels = gpconv_L2_output_dim,kernel_size = 2)
        self.query_tanh_L2 = nn.Tanh()
        self.query_conv_L3 = nn.Conv1d(in_channels = gpconv_L3_input_dim,out_channels = gpconv_L3_output_dim,kernel_size = 2)
        self.query_tanh_L3 = nn.Tanh()

        # doc conv
        self.doc_conv_L1 = nn.Conv1d(in_channels = gpconv_L1_input_dim,out_channels = gpconv_L1_output_dim,kernel_size = 2)
        self.doc_tanh_L1 = nn.Tanh()
        self.doc_conv_L2 = nn.Conv1d(in_channels = gpconv_L2_input_dim,out_channels = gpconv_L2_output_dim,kernel_size = 2)
        self.doc_tanh_L2 = nn.Tanh()
        self.doc_conv_L3 = nn.Conv1d(in_channels = gpconv_L3_input_dim,out_channels = gpconv_L3_output_dim,kernel_size = 2)
        self.doc_tanh_L3 = nn.Tanh()

        # mf model
        self.bilinear = nn.Bilinear(in1_features = bilin_input_dim,in2_features = bilin_input_dim,out_features = bilin_output_dim,bias = True)
        self.sigmoid = nn.Sigmoid()
        self.amp2d = nn.AdaptiveMaxPool2d(output_size = (mf_model_pool_dim,mf_model_pool_dim))

        # mfcnn
        self.mfcnn_conv_L1 = nn.Conv2d(in_channels = mfcnn_L1_input_dim,out_channels = mfcnn_L1_output_dim,kernel_size = 2,stride = 1,padding = 4)
        self.mfcnn_sgmd_L1 = nn.Sigmoid()
        self.mfcnn_pool_L1 = nn.AdaptiveMaxPool2d(output_size = (mfcnn_L1_pool_HW,mfcnn_L1_pool_HW))
        self.mfcnn_conv_L2 = nn.Conv2d(in_channels = mfcnn_L2_input_dim,out_channels = mfcnn_L2_output_dim,kernel_size = 2,stride = 1,padding = 4)
        self.mfcnn_sgmd_L2 = nn.Sigmoid()
        self.mfcnn_pool_L2 = nn.AdaptiveMaxPool2d(output_size = (mfcnn_L2_pool_HW,mfcnn_L2_pool_HW))
        self.mfcnn_conv_L3 = nn.Conv2d(in_channels = mfcnn_L2_input_dim,out_channels = mfcnn_L2_output_dim,kernel_size = 2,stride = 1,padding = 4)
        self.mfcnn_sgmd_L3 = nn.Sigmoid()
        self.mfcnn_pool_L3 = nn.AdaptiveMaxPool2d(output_size = (mfcnn_L3_pool_HW,mfcnn_L3_pool_HW))
        
        # mlp
        self.mlp_linear = nn.Linear(in_features = mfcnn_L3_pool_HW,out_features = 1)
        self.mlp_sigmoid = nn.Sigmoid()
        
    def conv_pooling(self,query,ki,query_index,layer_num):

        if query_index == 1: # query network
            pre_padding = Variable(torch.zeros(query.shape[0],4,query.shape[2])).to(device)
            back_padding = Variable(torch.zeros(query.shape[0],4,query.shape[2])).to(device)
            query = torch.cat((pre_padding,query),1)
            query = torch.cat((query,back_padding),1)
            query = query.transpose(1,2)

            if layer_num == 1:
                query_conv_output = self.query_tanh_L1(self.query_conv_L1(query))

            elif layer_num == 2:
                query_conv_output = self.query_tanh_L2(self.query_conv_L2(query))

            elif layer_num == 3:
                query_conv_output = self.query_tanh_L3(self.query_conv_L3(query))

            else:
                print('query layer_num not in 1-3')

        elif query_index == 2:  # doc network
            pre_padding = Variable(torch.zeros(query.shape[0],4,query.shape[2])).to(device)
            back_padding = Variable(torch.zeros(query.shape[0],4,query.shape[2])).to(device)
            query = torch.cat((pre_padding,query),1)
            query = torch.cat((query,back_padding),1)
            query = query.transpose(1,2)

            if layer_num == 1:
                query_conv_output = self.doc_tanh_L1(self.doc_conv_L1(query))

            elif layer_num == 2:
                query_conv_output = self.doc_tanh_L2(self.doc_conv_L2(query))

            elif layer_num == 3:
                query_conv_output = self.doc_tanh_L3(self.doc_conv_L3(query))

            else:
                print('doc layer_num not in 1-3')

        else:
            print('query index not in 1-2')

        query_pooling_output = k_max_pooling(query_conv_output,1,ki)

        return query_conv_output,query_pooling_output

    def mf_model(self,gpcnn_query,gpcnn_pos_doc):
        mf_matrix_pos = Variable(torch.zeros(gpcnn_query.shape[0],gpcnn_pos_doc.shape[0])).to(device)  # mf matrix
        for i in range(gpcnn_query.shape[0]):
            for j in range(gpcnn_pos_doc.shape[0]):
                u = gpcnn_query[i]
                v = gpcnn_pos_doc[j]
                bilin_output = self.sigmoid(self.bilinear(u,v))
                mf_matrix_pos[i][j] = bilin_output
        return mf_matrix_pos
        
    def mfcnn(self,mf_pos):
        # layer 1 conv + pooling
        mf_pos = mf_pos.reshape(1,1,mf_pos.shape[0],mf_pos.shape[1])
        mfcnn_L1_conv_output = self.mfcnn_sgmd_L1(self.mfcnn_conv_L1(mf_pos))
        mfcnn_L1_pool_output = self.mfcnn_pool_L1(mfcnn_L1_conv_output)
        mfcnn_L1_pool_output = k_max_pooling(mfcnn_L1_pool_output,1,1)
        
        # layer 2 conv + pooling
        mfcnn_L2_conv_output = self.mfcnn_sgmd_L2(self.mfcnn_conv_L2(mfcnn_L1_pool_output))
        mfcnn_L2_pool_output = self.mfcnn_pool_L2(mfcnn_L2_conv_output)
        mfcnn_L2_pool_output = k_max_pooling(mfcnn_L2_pool_output,1,1)
        
        # layer 3 conv + pooling
        mfcnn_L3_conv_output = self.mfcnn_sgmd_L3(self.mfcnn_conv_L3(mfcnn_L2_pool_output))
        mfcnn_L3_pool_output = self.mfcnn_pool_L3(mfcnn_L3_conv_output)
        mfcnn_L3_pool_output = k_max_pooling(mfcnn_L3_pool_output,1,1)
        
        return mfcnn_L3_pool_output.squeeze()
        
    
    def forward(self,query,pos_doc,neg_docs):
        gpcnn_query = k_max_pooling(query,2,gpconv_L3_output_dim)
        gpcnn_pos_doc = k_max_pooling(pos_doc,2,gpconv_L3_output_dim)
        gpcnn_neg_docs = [k_max_pooling(neg_doc,2,gpconv_L3_output_dim) for neg_doc in neg_docs]

        # sequence length |S|
        query_len = query.shape[1]
        pos_doc_len = pos_doc.shape[1]
        neg_docs_len = [neg_doc.shape[1] for neg_doc in neg_docs]

        # top k values [k1,k2,1] L = 3
        query_k = [max(4,math.ceil((3-1)*query_len/3)),max(4,math.ceil((3-2)*query_len/3)),1]
        pos_doc_k = [max(4,math.ceil((3-1)*pos_doc_len/3)),max(4,math.ceil((3-2)*pos_doc_len/3)),1]
        neg_docs_k = []
        for neg_doc_len in neg_docs_len:
            neg_doc_k = [max(4,math.ceil((3-1)*neg_doc_len/3)),max(4,math.ceil((3-2)*neg_doc_len/3)),1]
            neg_docs_k.append(neg_doc_k)

        # gpCNN 
        # layer 1 conv + pooling  query
        query_L1_conv_output,query_L1_pooling_output = self.conv_pooling(query,query_k[0],1,1) # short phrase representation

        # layer 2 conv + pooling
        query_L2_conv_output,query_L2_pooling_output = self.conv_pooling(query_L1_pooling_output,query_k[1],1,2) # long phrase representation

        # layer 3 conv + pooling
        query_L3_conv_output,query_L3_pooling_output = self.conv_pooling(query_L2_pooling_output,query_k[2],1,1) # sentence phrase representation

        # mf input  ===============================================================
        gpcnn_query = torch.cat((gpcnn_query,query_L1_pooling_output),1)
        gpcnn_query = torch.cat((gpcnn_query,query_L2_pooling_output),1)
        gpcnn_query = torch.cat((gpcnn_query,query_L3_pooling_output),1)

        #layer 1 conv + pooling  pos_doc
        pos_doc_L1_conv_output,pos_doc_L1_pooling_output = self.conv_pooling(pos_doc,pos_doc_k[0],2,1) # short phrase representation

        #layer 2 conv + pooling
        pos_doc_L2_conv_output,pos_doc_L2_pooling_output = self.conv_pooling(pos_doc_L1_pooling_output,pos_doc_k[1],2,2) # long phrase representation

        #layer 3 conv + pooling
        pos_doc_L3_conv_output,pos_doc_L3_pooling_output = self.conv_pooling(pos_doc_L2_pooling_output,pos_doc_k[2],2,3) # sentence phrase representation

        # mf input  =====================================================================
        gpcnn_pos_doc = torch.cat((gpcnn_pos_doc,pos_doc_L1_pooling_output),1)
        gpcnn_pos_doc = torch.cat((gpcnn_pos_doc,pos_doc_L2_pooling_output),1)
        gpcnn_pos_doc = torch.cat((gpcnn_pos_doc,pos_doc_L3_pooling_output),1)

        #layer 1 conv + pooling  neg_docs
        neg_docs_L1_pooling_output = []
        for i in range(neg_samples):
            neg_doc_L1_conv_output,neg_doc_L1_pooling_output = self.conv_pooling(neg_docs[i],neg_docs_k[i][0],2,1) # short phrase representation
            neg_docs_L1_pooling_output.append(neg_doc_L1_pooling_output)

        #layer 2 conv + pooling
        neg_docs_L2_pooling_output = []
        for i in range(neg_samples):
            neg_doc_L2_conv_output,neg_doc_L2_pooling_output = self.conv_pooling(neg_docs_L1_pooling_output[i],neg_docs_k[i][1],2,2) # long phrase representation
            neg_docs_L2_pooling_output.append(neg_doc_L2_pooling_output)

        #layer 3 conv + pooling
        neg_docs_L3_pooling_output = []
        for i in range(neg_samples):
            neg_doc_L3_conv_output,neg_doc_L3_pooling_output = self.conv_pooling(neg_docs_L2_pooling_output[i],neg_docs_k[i][2],2,3) # long phrase representation
            neg_docs_L3_pooling_output.append(neg_doc_L3_pooling_output)

        # mf input  ===================================================================================
        for i in range(neg_samples):
            gpcnn_neg_docs[i] = torch.cat((gpcnn_neg_docs[i],neg_docs_L1_pooling_output[i]),1)
            gpcnn_neg_docs[i] = torch.cat((gpcnn_neg_docs[i],neg_docs_L2_pooling_output[i]),1)
            gpcnn_neg_docs[i] = torch.cat((gpcnn_neg_docs[i],neg_docs_L3_pooling_output[i]),1)


        # mf model bilinear
        mf_matrix_pos = self.mf_model(gpcnn_query,gpcnn_pos_doc)  # mf matrix
        mf_matrix_negs = [self.mf_model(gpcnn_query,gpcnn_neg_docs[k]) for k in range(neg_samples)]
        
        
        # mf model dynamic 2d pooling
        mf_pos = (self.amp2d(mf_matrix_pos))
        mf_negs = [(self.amp2d(mf_matrix_neg)) for mf_matrix_neg in mf_matrix_negs]
                
        # mfCNN
        mfcnn_pos = self.mfcnn(mf_pos)
        mfcnn_negs = [self.mfcnn(mf_neg) for mf_neg in mf_negs]
        
        # mlp
        mlp_pos = [self.mlp_sigmoid(self.mlp_linear(self.mlp_sigmoid(self.mlp_linear(mfcnn_pos))))]
        mlp_negs = [self.mlp_sigmoid(self.mlp_linear(self.mlp_sigmoid(self.mlp_linear(mfcnn_neg)))) for mfcnn_neg in mfcnn_negs]

        s = mlp_pos + mlp_negs
        s = torch.stack(s)
        return s

model = multigrancnn().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(),lr = learning_rate)

for epoch in range(num_epochs):

    for i in range(batch_size):
        
        y = np.ndarray(1)
        y[0] = 0
        y = Variable(torch.from_numpy(y).long()).to(device)
        
        querys[i] = Variable(querys[i].float()).to(device)
        pos_docs[i] = Variable(pos_docs[i].float()).to(device)
        neg_docs[i] = [Variable(neg_docs[i][j].float()).to(device) for j in range(neg_samples)]
        
        # forward pass
        y_pred = model(querys[i],pos_docs[i],neg_docs[i])
        loss = criterion(y_pred.reshape(1,neg_samples+1),y)
        
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print loss
        print('epoch: {}, steps: {}, loss: {}'.format(epoch,i,loss.item()))
