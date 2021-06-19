import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import math
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.metrics import F1
import numpy as np
import glob

#torch.set_default_tensor_type('torch.cuda.FloatTensor')

'''
ntokens = 100  # the size of vocabulary
emsize = 768  # embedding dimension
nhid = emsize * 2  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 1  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # the number of heads in the multiheadattention models
dropout = 0.1  # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
'''


class PositionalEncoding(nn.Module): #PEの定義

    def __init__(self, d_model, dropout=0.1, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x, index):
        original_pe = torch.zeros(x.size(0), self.max_len, self.d_model)
        for i, tree in enumerate(index):
            for j, order in enumerate(tree):
                original_pe[i][j] = self.pe[order]
        x = x + original_pe[:, :x.size(1), :]

        return self.dropout(x)


class LitAutoEncoder(pl.LightningModule): #Transformer

    def __init__(self, ninp=768, nhead=8, nhid=768 * 2, nlayers=1, dropout=0.1):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.nhead = nhead
        self.decoder1 = nn.Linear(768, 500)
        self.relu = nn.ReLU()
        self.decoder2 = nn.Linear(500, 2)
        self.logsoftmax = torch.nn.Softmax(dim=1)
        self.f1 = F1(num_classes=2)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, src_mask = None, pe_index = None):
        #x = x.transpose(1, 0)
        self.src_mask = src_mask

        if self.src_mask is None or self.src_mask.size(0) != x.size(0):
            device = x.device
            mask = self._generate_square_subsequent_mask(x.size(0)).to(device)
            self.src_mask = mask

        if pe_index != None:
            x = self.pos_encoder(x, pe_index)
        '''
        if self.pos_type == "order":
            x = self.pos_encoder(x)
        elif self.pos_type == "depth":
            x = self.pos_encoder(x)
        '''
        embedding = self.transformer_encoder(x, mask=torch.cat([self.src_mask for _ in range(self.nhead)], dim=0))
        #embedding = self.transformer_encoder(x)
        #embedding = embedding.transpose(1, 0)
        embedding = torch.mean(embedding, 1)
        embedding = self.decoder1(embedding)
        embedding = self.relu(embedding)
        embedding = self.decoder2(embedding)
        embedding = self.logsoftmax(embedding)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y, m, p = batch
        y_hat = self.forward(x, m, pe_index=p)
        loss = F.nll_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, m, p = batch
        y_hat = self.forward(x, m, pe_index=p)
        print(y_hat)
        print(y)
        val_loss = F.nll_loss(y_hat, y)
        val_acc = pl.metrics.Accuracy()
        self.log('val_loss', val_loss)
        self.log('val_acc', val_acc(y_hat, y))#外部ライブラリを使うとGPU/CPUで変数衝突
        y_hat_b = []
        for i in y_hat:
            if (i[0] > i[1]):
                y_hat_b.append(0)
            else:
                y_hat_b.append(1)
        y_hat_b = torch.tensor(y_hat_b)
        f1 = (y_hat_b, y)
        self.log('f1', f1)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y, m, p = batch
        y_hat = self.forward(x, m, p)
        test_loss = F.nll_loss(y_hat, y)
        test_acc = pl.metrics.Accuracy()
        self.log('test_loss', test_loss)
        self.log('test_acc', test_acc(y_hat, y))
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.0001)
        return optimizer


# In[ ]:

import collections

def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def Tensor_Maker(dataset, max_node_num=300, pe_type=None):
    print("////////////////////0000 s////////////////////")
    data = []
    target = []
    mask = []
    pe = []
    print("////////////////////0000 e////////////////////")

    for tweets in dataset:
        print("////////////////////0001 s////////////////////")
        # kaku Reply Tree
        tweets_texts = [[0 for _ in range(768)] for _ in range(max_node_num)]
        tweets_pes = [0 for _ in range(max_node_num)]
        int_num_dict = {}
        print("////////////////////0001-1////////////////////")

        if (tweets[0][0] == 1):
            target.append(1)
        elif (tweets[0][0] == -1):
            target.append(0)

        print("////////////////////0001-2////////////////////")

        # train_target.append(i[0][0])
        for index, a_tweet in enumerate(tweets):
            print("////////////////////0002 s////////////////////")
            print("index : " + str(index)) #+ ", j : " + str(j))
            # kaku tweets
            if(index >= max_node_num):break

            tweets_texts[index] = a_tweet[3].tolist() #ツイートのベクトルを追加

            if(pe_type == "Order"):
                tweets_pes[index] = index
            if(pe_type == "Time"):
                tweets_pes[index] = a_tweet[1]
            elif(pe_type == "Depth"):
                tweets_pes[index] = a_tweet[2]

            int_num_dict[a_tweet[4]] = index #ツイートのIDをINDEX化
            print("////////////////////0002 e////////////////////")

        print("////////////////////0001-3////////////////////")

        G = [[] for i in range(len(tweets))]
        print(len(tweets))
        for a_tweet in tweets: #枝の情報を構成
            print("////////////////////0003 s////////////////////")
            if(a_tweet[5]!=None):
                G[int_num_dict[a_tweet[4]]].append(int_num_dict[a_tweet[5]])
            print("////////////////////0003 e////////////////////")

        print("////////////////////0001-4////////////////////")
        # dfs
        def dfs(v):
            if (temp[v] == False): return  # 同じ頂点を2度以上調べないためのreturn
            temp[v] = False
            for vv in G[v]: dfs(vv)

        print("////////////////////0001-5////////////////////")

        temps = []

        print("////////////////////0001-6////////////////////")

        for a_tweet in range(max_node_num): #行番目のノードが列番目のノードに到達可能かを判定
            temp = [True] * max_node_num
            if(a_tweet < len(tweets)):
                dfs(a_tweet)
            else:
                temp = [False] * max_node_num
            temps.append(temp)

        print("////////////////////0001-7////////////////////")

        data.append(tweets_texts)
        mask.append(temps)
        pe.append(tweets_pes)

        print("////////////////////0001 e////////////////////")

    data = list(flatten(data))
    data = np.array(data)
    target = np.array(target)
    mask = np.array(mask)
    pe = np.array(pe)

    data = torch.from_numpy(data).float()
    target = torch.from_numpy(target)
    mask = torch.from_numpy(mask)
    pe = torch.from_numpy(pe)

    data = data.view(-1, max_node_num, 768)
    print(data.size())
    print(target.size())
    print(mask.size())
    print(pe.size())

    return data, target, mask, pe


#TODO datasetsにMASKを入力し，各ステップごとのbatchの返数にMASKを追加しforward()に渡す

def experiment(pe_type):
    for i in range(5):
        torch.manual_seed(i)

        train_data, train_target, train_mask, train_pe = Tensor_Maker(train_dataset, pe_type=pe_type)
        val_data, val_target, val_mask, val_pe = Tensor_Maker(val_dataset, pe_type=pe_type)
        test_data, test_target, test_mask, test_pe = Tensor_Maker(test_dataset, pe_type=pe_type)

        T_data = torch.transpose(train_data, 0, 1)
        T_data = T_data[:300]
        T_data = torch.transpose(T_data, 0, 1)
        train_pe = torch.transpose(train_pe, 0, 1)
        train_pe = train_pe[:300]
        train_pe = torch.transpose(train_pe, 0, 1)

        V_data = torch.transpose(val_data, 0, 1)
        V_data = V_data[:300]
        V_data = torch.transpose(V_data, 0, 1)
        val_pe = torch.transpose(val_pe, 0, 1)
        val_pe = val_pe[:300]
        val_pe = torch.transpose(val_pe, 0, 1)

        TE_data = torch.transpose(test_data, 0, 1)
        TE_data = TE_data[:300]
        TE_data = torch.transpose(TE_data, 0, 1)
        test_pe = torch.transpose(test_pe, 0, 1)
        test_pe = test_pe[:300]
        test_pe = torch.transpose(test_pe, 0, 1)

        train_datasets = torch.utils.data.TensorDataset(T_data, train_target, train_mask, train_pe)
        val_datasets = torch.utils.data.TensorDataset(V_data, val_target, val_mask, val_pe)
        test_datasets = torch.utils.data.TensorDataset(TE_data, test_target, test_mask, test_pe)

        #Train

        autoencoder = LitAutoEncoder()
        early_stopping = EarlyStopping('val_loss')
        #print(autoencoder.decoder2.weight, autoencoder.decoder2.bias)

        trainer = pl.Trainer(
            max_epochs=10000,
            callbacks = [early_stopping] #,
            #gpus = 1,
            #accelerator='dp'
        )

        DL_Train = DataLoader(train_datasets, batch_size=8, shuffle=True)
        DL_Val = DataLoader(val_datasets, batch_size=1, shuffle=True)
        DL_Test = DataLoader(test_datasets, shuffle=True)

        trainer.fit(
            autoencoder,
            DL_Train,
            DL_Val
        )
        trainer.test(
            test_dataloaders=DL_Train
        )

# Dataload
Tweetslist = []

for filename in glob.glob("DATA/Seq_Tfm_2/Pos*.npy"):
    a = np.load(filename, allow_pickle=True)  # .tolist()
    Tweetslist.append(a)

for filename in glob.glob("DATA/Seq_Tfm_2/Neg*.npy"):
    a = np.load(filename, allow_pickle=True)  # .tolist()
    Tweetslist.append(a)

np.random.shuffle(Tweetslist)
tweets = np.array(Tweetslist, dtype=object)
print(type(tweets[0]))

# In[ ]:

print(tweets.shape)

# In[ ]:

n_samples = len(tweets)  # n_samples is 60000
t2_size = int(len(tweets) * 0.8)  # train_size is 48000
val_size = n_samples - t2_size

t2_dataset, val_dataset = torch.utils.data.random_split(tweets, [t2_size, val_size])

t2_samples = len(t2_dataset)
train_size = int(len(t2_dataset) * 0.8)  # train_size is 48000
val_size = t2_samples - train_size

train_dataset, test_dataset = torch.utils.data.random_split(t2_dataset, [train_size, val_size])

print(len(train_dataset))
print(len(val_dataset))
print(len(test_dataset))

experiment('Order')
experiment('Depth')
experiment('Time')
