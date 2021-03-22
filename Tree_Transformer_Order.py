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


# # Data Scraping

# In[ ]:


import pickle
import glob
import re


for filename in glob.glob("/home/acc13097es/DATA/Flaming_SCRAPED/*.txt"):
    f = open(filename,"rb")
    df = pickle.load(f)
    tid = f.name[30:]
    tid = re.sub("\\D", "", tid)
    f.close()
    
    txtpos = open("/home/acc13097es/DATA/Seq_Tfm/Pos" + tid + ".txt", "w")
    
    for i in df.itertuples():
        print(i[4])
        txtpos.write(i[4] + '\n')
        
    txtpos.close()



for filename in glob.glob("/home/acc13097es/DATA/NotFlaming_SCRAPED/*.txt"):
    f = open(filename,"rb")
    df = pickle.load(f)
    tid = f.name[30:]
    tid = re.sub("\\D", "", tid)
    f.close()
    
    txtneg = open("/home/acc13097es/DATA/Seq_Tfm/Neg" + tid + ".txt", "w")
    
    for i in df.itertuples():
        print(i[4])
        txtneg.write(i[4] + '\n')
        
    txtneg.close()


# In[ ]:


get_ipython().run_line_magic('pip', 'install pytorch-pretrained-bert')


# In[ ]:


get_ipython().system('pip install transformers')


# In[ ]:


get_ipython().system('git clone https://github.com/huggingface/transformers')
get_ipython().run_line_magic('cd', 'transformers')
get_ipython().system('pip install .')


# In[ ]:


get_ipython().system('apt install aptitude')
get_ipython().system('aptitude install mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file -y')
get_ipython().system('pip install mecab-python3==0.7')
get_ipython().system('pip install fugashi ipadic')


# In[ ]:


import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM


# In[ ]:


tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

text = '父の父は、祖父。'
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)


# In[ ]:


# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 2
tokenized_text[masked_index] = '[MASK]'
print(tokenized_text)

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
print(indexed_tokens)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
print(tokens_tensor)


# In[ ]:


from transformers import BertJapaneseTokenizer, BertModel
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model_bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model_bert.eval()


# In[ ]:


import numpy as np

def calc_embedding(text):
    bert_tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + bert_tokens[:126] + ["[SEP]"])
    tokens_tensor = torch.tensor(ids).reshape(1, -1)

    with torch.no_grad():
        output = model_bert(tokens_tensor)

    return output[1].numpy()


# In[ ]:


text = '私はからあげです'
print(type(calc_embedding(text)))
b = np.array([[0], [53]])
np.append(b, [calc_embedding(text)])
print(calc_embedding(text))


# v Probably Unneeded

# In[ ]:


import pickle
import glob
import re


for filename in glob.glob("/home/acc13097es/DATA/Flaming_SCRAPED/*.txt"):
    f = open(filename,"rb")
    df = pickle.load(f)
    tid = f.name[30:]
    tid = re.sub("\\D", "", tid)
    f.close()
    
    a = np.zeros((0,768))
    
    for i in df.itertuples():
        print(i[4])
        b = calc_embedding(i[4]) 
        a = np.r_[a,b.reshape(1,-1)]
        
    print(type(a))
    print(a.shape)
    np.save("/home/acc13097es/DATA/Seq_Tfm_2/Pos" + tid, a)
    

for filename in glob.glob("/home/acc13097es/DATA/NotFlaming_SCRAPED/*.txt"):
    f = open(filename,"rb")
    df = pickle.load(f)
    tid = f.name[30:]
    tid = re.sub("\\D", "", tid)
    f.close()
    
    a = np.zeros((0,768))
    
    for i in df.itertuples():
        print(i[4])
        b = calc_embedding(i[4]) 
        a = np.r_[a,b.reshape(1,-1)]
        
    print(type(a))
    print(a.shape)
    np.save("/home/acc13097es/DATA/Seq_Tfm_2/Neg" + tid, a)
    
    
for filename in glob.glob("/home/acc13097es/DATA/UnLabeled_SCRAPED/*.txt"):
    f = open(filename,"rb")
    df = pickle.load(f)
    tid = f.name[30:]
    tid = re.sub("\\D", "", tid)
    f.close()
    
    a = np.zeros((0,768))
    
    for i in df.itertuples():
        print(i[4])
        b = calc_embedding(i[4]) 
        a = np.r_[a,b.reshape(1,-1)]
        
    print(type(a))
    print(a.shape)
    np.save("/home/acc13097es/DATA/Seq_Tfm_2/UnL" + tid, a)


# **DO THIS**

# In[ ]:


def SearchRootNum(reply_to_id):
    if(reply_to_id == None):
        return 0
    a = df[df['id']==reply_to_id].index.tolist()
    if(a == []):
        return 0
    #print(a)
    a_se = df.loc[a[0]]
    
    return SearchRootNum(a_se['reply_to_id']) + 1


# In[ ]:


import pickle
import glob
import re
import math


for filename in glob.glob("/home/acc13097es/DATA/Flaming_SCRAPED/*.txt"):
    f = open(filename,"rb")
    df = pickle.load(f)
    tid = f.name[30:]
    tid = re.sub("\\D", "", tid)
    f.close()
    
    a = np.zeros((0,4), dtype = object)
    
    roottime = None
    
    for i in df.itertuples():
        if(tid == i[1]): roottime = i[8]
        
        td = i[8] - roottime
        sec = td.total_seconds()
        hours = math.ceil(sec / 3600)
        
        b = np.array([1, hours, SearchRootNum(i[3]), calc_embedding(i[4])])
        a = np.r_[a,b.reshape(1,-1)]
        
    print(a)
    print(a.shape)
    np.save("/home/acc13097es/DATA/Seq_Tfm_2/Pos" + tid, a)
    

for filename in glob.glob("/home/acc13097es/DATA/NotFlaming_SCRAPED/*.txt"):
    f = open(filename,"rb")
    df = pickle.load(f)
    tid = f.name[30:]
    tid = re.sub("\\D", "", tid)
    f.close()
    
    a = np.zeros((0,4), dtype = object)
    
    roottime = None
    
    for i in df.itertuples():
        if(tid == i[1]): roottime = i[8]
        
        td = i[8] - roottime
        sec = td.total_seconds()
        hours = math.ceil(sec / 3600)
        
        b = np.array([-1, hours, SearchRootNum(i[3]), calc_embedding(i[4])])
        a = np.r_[a,b.reshape(1,-1)]
        
    print(a.shape)
    np.save("/home/acc13097es/DATA/Seq_Tfm_2/Neg" + tid, a)

'''
for filename in glob.glob("/home/acc13097es/DATA/UnLabeled_SCRAPED/*.txt"):
    f = open(filename,"rb")
    df = pickle.load(f)
    tid = f.name[30:]
    tid = re.sub("\\D", "", tid)
    f.close()
    
    a = np.zeros((0,4), dtype = object)
    
    roottime = None
    
    try:
        for i in df.itertuples():
            if(tid == i[1]): roottime = i[8]

            #print(tid + ' ' + i[1])

            td = i[8] - roottime
            sec = td.total_seconds()
            hours = math.ceil(sec / 3600)

            b = np.array([0, hours, SearchRootNum(i[3]), calc_embedding(i[4])])
            a = np.r_[a,b.reshape(1,-1)]

        print(a.shape)
        np.save("/home/acc13097es/DATA/Seq_Tfm_2/UnL" + tid, a)
        
    except:
        print('err')
        continue
        
'''


# In[ ]:


for filename in glob.glob("/home/acc13097es/DATA/UnLabeled_SCRAPED/*.txt"):
    f = open(filename,"rb")
    df = pickle.load(f)
    tid = f.name[30:]
    tid = re.sub("\\D", "", tid)
    f.close()
    
    for i in df.itertuples():
            
        print(i[8])


# # Learning

# In[ ]:


Tweetslist = []

for filename in glob.glob("/home/acc13097es/DATA/Seq_Tfm_2/Pos*.npy"):
    a = np.load(filename, allow_pickle = True) #.tolist()
    Tweetslist.append(a)
    
for filename in glob.glob("/home/acc13097es/DATA/Seq_Tfm_2/Neg*.npy"):
    a = np.load(filename, allow_pickle = True) #.tolist()
    Tweetslist.append(a)
    

np.random.shuffle(Tweetslist)
tweets = np.array(Tweetslist, dtype=object)
print(type(tweets[0]))


# In[ ]:


print(tweets.shape)


# In[ ]:


n_samples = len(tweets) # n_samples is 60000
t2_size = int(len(tweets) * 0.8) # train_size is 48000
val_size = n_samples - t2_size 

t2_dataset, val_dataset = torch.utils.data.random_split(tweets, [t2_size, val_size])

t2_samples = len(t2_dataset) 
train_size = int(len(t2_dataset) * 0.8) # train_size is 48000
val_size = t2_samples - train_size 

train_dataset, test_dataset = torch.utils.data.random_split(t2_dataset, [train_size, val_size])

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))


# In[ ]:


import collections

def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
            
#---------train------------------------------
#---------------------------------------------
train_data = []
train_target = []

for index_i, i  in enumerate(train_dataset):
    #kaku Reply Tree
    a =  [[0 for _ in range(768)] for _ in range(500)]
    
    if(i[0][0] == 1):
        train_target.append(1)
    elif(i[0][0] == -1):
        train_target.append(0)
    
    #train_target.append(i[0][0])
    '''
    for index, j in enumerate(i):
        #kaku tweets
        #print(len(a[index]))
        for index_k, k in enumerate(j[3].tolist()):
            #a[index] = j[3].tolist()
            a[index][index_k] = k
    '''
    for index, j in enumerate(i):
        #kaku tweets
        #print(len(a[index]))
        a[index] = j[3].tolist()
        
    train_data.append(a)
    
train_data = list(flatten(train_data))

            
train_data = np.array(train_data)
train_target = np.array(train_target)

train_data = torch.from_numpy(train_data).float()
train_target = torch.from_numpy(train_target)

train_data = train_data.view(-1, 500, 768)
print(train_data.size())
print(train_target.size())


#---------val------------------------------
#---------------------------------------------
val_data = []
val_target = []

for i in val_dataset:
    #kaku Reply Tree
    a =  [[0 for _ in range(768)] for _ in range(500)]
    
    if(i[0][0] == 1):
        val_target.append(1)
    elif(i[0][0] == -1):
        val_target.append(0)
    
    #train_target.append(i[0][0])
    for index, j in enumerate(i):
        #kaku tweets
        #print(len(a[index]))
        a[index] = j[3].tolist()
        
    val_data.append(a)
    
val_data = list(flatten(val_data))

            
val_data = np.array(val_data)
val_target = np.array(val_target)

val_data = torch.from_numpy(val_data).float()
val_target = torch.from_numpy(val_target)

val_data = val_data.view(-1, 500, 768)
print(val_data.size())
print(val_target.size())


#---------test-------------------------------
#---------------------------------------------
test_data = []
test_target = []

for i in test_dataset:
    #kaku Reply Tree
    a =  [[0 for _ in range(768)] for _ in range(500)]
    
    if(i[0][0] == 1):
        test_target.append(1)
    elif(i[0][0] == -1):
        #Flaming=1, UnFlaming=0
        test_target.append(0)
    
    #train_target.append(i[0][0])
    for index, j in enumerate(i):
        #kaku tweets
        #print(len(a[index]))
        a[index] = j[3].tolist()
        
    test_data.append(a)
    
test_data = list(flatten(test_data))

            
test_data = np.array(test_data)
test_target = np.array(test_target)

test_data = torch.from_numpy(test_data).float()
test_target = torch.from_numpy(test_target)

test_data = test_data.view(-1, 500, 768)
print(test_data.size())
print(test_target.size())

'''
val_data = []
val_target = []

for i in val_dataset:
    #kaku Reply Tree
    a = []
    val_target.append(i[0][0])
    for j in i:
        #kaku tweets
        a.extend(j[3])
    
    b = np.array(a, dtype = object)
    val_data.append(b)
            
val_data = np.array(val_data, dtype = object)
val_target = np.array(val_target, dtype = object)

val_data = torch.from_numpy(val_data)
val_target = torch.from_numpy(val_target)
print(len(val_data))


test_data = []
test_target = []

for i in test_dataset:
    #kaku Reply Tree
    a = []
    test_target.append(i[0][0])
    for j in i:
        #kaku tweets
        a.extend(j[3])
    
    b = np.array(a, dtype = object)
    test_data.append(b)
            
test_data = np.array(test_data, dtype = object)
test_target = np.array(test_target, dtype = object)

test_data = torch.from_numpy(test_data)
test_target = torch.from_numpy(test_target)
print(len(test_data))
'''


# In[ ]:


print(i[0][0])


# In[ ]:


UnLabeledData = []
for filename in glob.glob("/home/acc13097es/DATA/Seq_Tfm_2/UnL*.npy"):
    a = np.load(filename, allow_pickle = True) #.tolist()
    UnLabeledData.append(a)
    
np.random.shuffle(UnLabeledData)
tweets = np.array(UnLabeledData, dtype=object)
print(type(tweets[0]))


UnLabeled_train_data = []

for index_i, i  in enumerate(UnLabeledData):
    #kaku Reply Tree
    a =  [[0 for _ in range(768)] for _ in range(500)]

    for index, j in enumerate(i):
        #kaku tweets
        #print(len(a[index]))
        a[index] = j[3].tolist()
        
    UnLabeled_train_data.append(a)
    
UnLabeled_train_data = list(flatten(UnLabeled_train_data))

            
UnLabeled_train_data = np.array(UnLabeled_train_data)

UnLabeled_train_data = torch.from_numpy(UnLabeled_train_data).float()

UnLabeled_train_data = UnLabeled_train_data.view(-1, 500, 768)
print(UnLabeled_train_data.size())


# In[ ]:


print(train_target)


# In[ ]:


def onehot(image_tensor, n_classes):
    h, w = image_tensor.size()[0], 1
    onehot = torch.LongTensor(n_classes, h, w).zero_()
    onehot = onehot.squeeze_(2)
    print(onehot)
    image_tensor = image_tensor.unsqueeze(0)
    onehot = onehot.scatter_(0, image_tensor, 1)
    return onehot

#--------------------------------------------------------
n_classes = 2
#--------------------------------------------------------

T_data = train_data
T_target = train_target

print(T_target)
'''
T_target = onehot(T_target, n_classes)
T_target = torch.transpose(T_target, 0, 1)
'''
print(T_target)

#--------------------------------------------------------
V_data = val_data
V_target = val_target

print(V_target)
'''
V_target = onehot(V_target, n_classes)
V_target = torch.transpose(V_target, 0, 1)
'''
print(T_target)

#--------------------------------------------------------
TE_data = test_data
TE_target = test_target

print(TE_target)
'''
TE_target = onehot(TE_target, n_classes)
TE_target = torch.transpose(TE_target, 0, 1)
'''
print(TE_target)


# In[ ]:


#torch.save(UnLabeled_train_data, '/home/acc13097es/T_T_DATA/UnLabeled_train_data.pt')

torch.save(T_data, '/home/acc13097es/T_T_DATA/T_data.pt')
torch.save(T_target, '/home/acc13097es/T_T_DATA/T_target.pt')
torch.save(V_data, '/home/acc13097es/T_T_DATA/V_data.pt')
torch.save(V_target, '/home/acc13097es/T_T_DATA/V_target.pt')
torch.save(TE_data, '/home/acc13097es/T_T_DATA/TE_data.pt')
torch.save(TE_target, '/home/acc13097es/T_T_DATA/TE_target.pt')


# In[ ]:


print(UnLabeled_train_data)


# # kokokara

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
writer = SummaryWriter(log_dir="/home/acc13097es/logs")

T_data  = torch.load('/home/acc13097es/T_T_DATA/T_data.pt')
T_target  = torch.load('/home/acc13097es/T_T_DATA/T_target.pt')
V_data  = torch.load('/home/acc13097es/T_T_DATA/V_data.pt')
V_target  = torch.load('/home/acc13097es/T_T_DATA/V_target.pt')
TE_data  = torch.load('/home/acc13097es/T_T_DATA/TE_data.pt')
TE_target  = torch.load('/home/acc13097es/T_T_DATA/TE_target.pt')

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
    
    #for h in range(0,5):
    for batch, i in enumerate(range(0, T_data.size(0) - 1, bptt)):
        T_data = T_data.to(device)
        #UnLabeled_train_data = UnLabeled_train_data.to(device)

        data =  get_batch(T_data, i)
        targets = get_batch(T_target, i)
        
        p = np.random.permutation(len(data))
        print(p)
        
        data = data[p]
        targets = targets[p]
        
        print(targets)
        
        data =  data.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        
        output = model(data)
        #print(output)
        #print(targets)

        loss = criterion(output, targets)

        
        
        #print(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        
        count += 1
        #TensorBoard
        writer.add_scalar("train", total_loss / (len(T_target)-1), epoch*5 + count)
        
        print("count = " + str(epoch*5 + count))        
        
        log_interval = 1
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


# In[ ]:


get_ipython().system(' tensorboard --logdir=logs  --port 6006 --host 0.0.0.0')


# In[13]:


test_loss = evaluate(best_model, TE_data,  TE_target)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)


# # Semi-Supervised

# In[ ]:


criterion = nn.CrossEntropyLoss()
lr = 3.0 # 学習率
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

import time
def train(model, T_data, T_target, UnLabeled_train_data):
    model.train() # 訓練モードに
    total_loss = 0.
    start_time = time.time()
    ntokens = 500
    
    '''
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    '''
    model.to(device)
    
    T_data = T_data.to(device)
    UnLabeled_train_data = UnLabeled_train_data.to(device)
            
    for h in range(0,5):
        for batch, i in enumerate(range(0, T_data.size(0) - 1, bptt)):
            print(T_target)
            
            data =  get_batch(T_data, i)
            targets = get_batch(T_target, i)
            data =  data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            print(output)
            print(targets)
            
            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
        
        UL_data = get_batch(UnLabeled_train_data, i)
        UL_data = UL_data.to(device)
        UL_output = model(UL_data)
        count = 0
        #print(UL_output)
        for index, result in enumerate(UL_output):
            if(result[0] >= 0.9):
                T_data = torch.cat((T_data, UL_data[index].unsqueeze(dim = 0)), axis = 0)
                #print('0:', result)
                #print(UL_data[index].unsqueeze(dim = 0))
                T_target = torch.cat((T_target, torch.tensor([0])), axis = 0)
                UnLabeled_train_data = torch.cat((UnLabeled_train_data[:i + index], UnLabeled_train_data[i + index + 1:]), axis = 0)
                count += 1
                #print(T_target)
                continue
            elif(result[1]>=0.9):
                T_data = torch.cat((T_data, UL_data[index].unsqueeze(dim = 0)), axis = 0)
                #print('1:', result)
                T_target = torch.cat((T_target, torch.tensor([1])), axis = 0)
                UnLabeled_train_data = torch.cat((UnLabeled_train_data[:i + index], UnLabeled_train_data[i + index + 1:]), axis = 0)
                count += 1
                #print(T_target)
                continue
        #####print(count)
        #print(T_data.size())
        #print(T_target.size())

        if(count <= 1):
            break
        

def evaluate(eval_model, val_data,  val_target):
    eval_model.eval() # 検証モードに
    total_loss = 0.
    ntokens = 500
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


# In[ ]:


best_val_loss = float("inf")
epochs = 10 # The number of epochs
best_model = None


for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model, T_data, T_target, UnLabeled_train_data)
    val_loss = evaluate(model, V_data,  V_target)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()


# In[ ]:


test_loss = evaluate(best_model, TE_data,  TE_target)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)


# In[ ]:


max_len=5000
dropout=0.1
dim = 768

def PE(position = 0):
    
    max_len=5000
    dropout=0.1
    dim = 768
    dropout = nn.Dropout(p=dropout)
    position = int(position)

    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    #register_buffer('pe', pe)
    R = np.array(pe)
    print(type(R))
    print(R[1])

    return R[position]


# In[ ]:


a = torch.tensor([0,1,2,3,4])
b = torch.tensor([0,1,2,3,4])
print(a)

for i in a:
    a[i] = a[i] + b[i]
    
    
print(a)


# In[ ]:


max_len=5000
dropout=0.1
dim = 768
dropout = nn.Dropout(p=dropout)

pe = torch.zeros(max_len, dim)
position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
pe = pe.unsqueeze(0).transpose(0, 1)

print(pe.size())


# In[ ]:




