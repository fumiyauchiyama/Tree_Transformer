#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import glob
import re
import math
import numpy as np
from torch.utils.tensorboard import SummaryWriter


# In[2]:

torch.manual_seed(26)


# In[3]:


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        ##self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder1 = nn.Linear(768, 500)
        self.relu = nn.ReLU()
        self.decoder2 = nn.Linear(500, 2)
        self.softmax = torch.nn.LogSoftmax(dim = 1)
        #self.softmax = torch.nn.Softmax(dim = 1)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder2.bias.data.zero_()
        self.decoder2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        
        src = src.transpose(1, 0)
        
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
            self.src_mask = mask

        #src = self.encoder(src) * math.sqrt(self.ninp)#embedding dimention
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        print(output.size())
        #output = output.view(-1, 768 * 500)
        
        output = output.transpose(1, 0)
        
        output = torch.mean(output, 1)
        print(output.size())
        output = self.decoder1(output)
        output = self.relu(output)
        #print(output.size())
        output = self.decoder2(output)
        #print(output.size())
        output = self.softmax(output)
        print(output.size())
        return output


# In[4]:


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# In[10]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


print("GPU環境：", device)  # GPU環境であれば、cuda と表示されます



# kokokara

# In[5]:


ntokens = 100 # the size of vocabulary
emsize = 768 # embedding dimension
nhid = 16 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 4 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8 # the number of heads in the multiheadattention models
dropout = 0.1 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout) #.to(device)


# In[6]:


#UnLabeled_train_data  = torch.load('/home/acc13097es/T_T_DATA/UnLabeled_train_data.pt')
writer = SummaryWriter(log_dir="logs")

T_data  = torch.load('T_T_DATA/T_data.pt')
T_target  = torch.load('T_T_DATA/T_target.pt')
V_data  = torch.load('T_T_DATA/V_data.pt')
V_target  = torch.load('T_T_DATA/V_target.pt')
TE_data  = torch.load('T_T_DATA/TE_data.pt')
TE_target  = torch.load('T_T_DATA/TE_target.pt')

T_data = torch.transpose(T_data, 0,1)
T_data  = T_data[:100]
T_data = torch.transpose(T_data, 0,1)

V_data = torch.transpose(V_data, 0,1)
V_data  = V_data[:100]
V_data = torch.transpose(V_data, 0,1)


TE_data = torch.transpose(TE_data, 0,1)
TE_data  = TE_data[:100]
TE_data = torch.transpose(TE_data, 0,1)


# In[7]:


bptt = 5
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    return data


# # Supervised

# In[8]:


criterion = nn.NLLLoss()
lr = 0.001 # 学習率
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


import time
def train(model, T_data, T_target, UnLabeled_train_data, epoch, count):
    model.train() # 訓練モードに
    total_loss = 0.
    start_time = time.time()
    ntokens = 100
    
    '''
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    '''
    model.to(device)
    T_data = T_data.to(device)
    T_target = T_target.to(device)

    p = np.random.permutation(len(T_data))

    T_data = T_data[p]
    T_target = T_target[p]
    
    #for h in range(0,5):
    for batch, i in enumerate(range(0, T_data.size(0) - 1, bptt)):

        #UnLabeled_train_data = UnLabeled_train_data.to(device)

        data =  get_batch(T_data, i)
        targets = get_batch(T_target, i)
        
        data =  data.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        #print(output)
        #print(targets)

        loss = criterion(output, targets)
        writer.add_scalar("train_loss", loss, epoch * 5 + count)

        
        
        #print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        
        count += 1
        #TensorBoard
        writer.add_scalar("train", total_loss / (len(T_target)-1), epoch*5 + count)
        
        print("count = " + str(epoch*5 + count))        
        
        log_interval = 4
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            writer.add_scalar("train_cur", cur_loss, epoch*5 + count)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(T_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, val_data,  val_target):
    eval_model.eval() # 検証モードに
    total_loss = 0.
    ntokens = 100
    with torch.no_grad():
        for i in range(0, val_data.size(0) - 1, bptt):
            data = val_data.to(device)
            targets = val_target.to(device)
            output = eval_model(data)
            
            print(output)
            print(targets)
            #output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(val_target) - 1)


# In[11]:


best_val_loss = float("inf")
epochs = 30 # The number of epochs
best_model = None

count = 0

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model, T_data, T_target, None, epoch, count)
    val_loss = evaluate(model, V_data,  V_target)
    
    #TensorBoard
    writer.add_scalar("val", val_loss, epoch)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)


    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()


# In[12]:


writer.close()


# In[13]:


test_loss = evaluate(best_model, TE_data,  TE_target)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
