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


class LitAutoEncoder(pl.LightningModule): #Transformer

    def __init__(self, ninp=768, nhead=8, nhid=768 * 2, nlayers=1, dropout=0.1, pos_type=None):
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.decoder1 = nn.Linear(768, 500)
        self.relu = nn.ReLU()
        self.decoder2 = nn.Linear(500, 2)
        self.logsoftmax = torch.nn.Softmax(dim=1)
        self.f1 = F1(num_classes=2)
        self.pos_type = pos_type

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, src_mask = None):
        x = x.transpose(1, 0)
        self.src_mask = src_mask

        if self.src_mask is None or self.src_mask.size(0) != x.size(0):
            device = x.device
            mask = self._generate_square_subsequent_mask(x.size(0)).to(device)
            self.src_mask = mask


        if self.pos_type == "order": #ここは将来的に時間と深さのPEを入れるかどうかを引数で指定できるようにしたいと思っています
            x = self.pos_encoder(x)
        '''
        if self.pos_type == "order":
            x = self.pos_encoder(x)
        elif self.pos_type == "depth":
            x = self.pos_encoder(x)
        '''
        embedding = self.transformer_encoder(x, self.src_mask)
        # embedding = self.transformer_encoder(x)
        embedding = embedding.transpose(1, 0)

        embedding = torch.mean(embedding, 1)
        embedding = self.decoder1(embedding)
        embedding = self.relu(embedding)
        embedding = self.decoder2(embedding)
        embedding = self.logsoftmax(embedding)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.nll_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # print(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        val_loss = F.nll_loss(y_hat, y)
        val_acc = pl.metrics.Accuracy()
        print(y_hat)
        print(y)
        self.log('val_loss', val_loss)
        self.log('val_acc', val_acc(y_hat, y))
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
        x, y = batch
        y_hat = self.forward(x)
        test_loss = F.nll_loss(y_hat, y)
        test_acc = pl.metrics.Accuracy()
        self.log('test_loss', test_loss)
        self.log('test_acc', test_acc(y_hat, y))
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.0001)
        return optimizer


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

# In[ ]:

import collections

def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def Tensor_Maker(dataset):
    data = []
    target = []
    mask = []

    for i in dataset:
        # kaku Reply Tree
        a = [[0 for _ in range(768)] for _ in range(200)]
        int_num_dict = {}

        if (i[0][0] == 1):
            target.append(1)
        elif (i[0][0] == -1):
            target.append(0)

        # train_target.append(i[0][0])
        for index, j in enumerate(i):
            # kaku tweets
            if(index >= 200):return
            a[index] = j[3].tolist() #ツイートのベクトルを追加
            int_num_dict[j[4]] = index #ツイートのIDをINDEX化

        G = [[] for i in range(len(i))]
        for j in i: #枝の情報を構成
            if(j[5]!=None):
                G[int_num_dict[j[4]]].append(int_num_dict[j[5]])

        # dfs
        def dfs(v):
            if (temp[v] == False): return  # 同じ頂点を2度以上調べないためのreturn
            temp[v] = False
            for vv in G[v]: dfs(vv)

        temps = []

        for j in range(200): #行番目のノードが列番目のノードに到達可能かを判定
            temp = [True] * 200
            if(j < len(i)):
                dfs(j)
            temps.append(temp)


        data.append(a)
        mask.append(temps)

    data = list(flatten(data))
    data = np.array(data)
    target = np.array(target)
    mask = np.array(mask)

    data = torch.from_numpy(data).float()
    target = torch.from_numpy(target)
    mask = torch.from_numpy(mask)

    data = data.view(-1, 200, 768)
    print(data.size())
    print(target.size())
    print(mask.size())

    return data, target, mask

train_data, train_target, train_mask = Tensor_Maker(train_dataset)
val_data, val_target, val_mask = Tensor_Maker(val_dataset)
test_data, test_target, test_mask = Tensor_Maker(test_dataset)

#TODO Tensor_Maker()でMASKを作り返せるようにする
#TODO datasetsにMASKを入力し，各ステップごとのbatchの返数にMASKを追加しforward()に渡す

def experiment(dataname, pos_bool):
    for i in range(5):
        torch.manual_seed(i)

        T_data = torch.load('T_' + dataname + '_DATA/T_data.pt')
        T_target = torch.load('T_' + dataname + '_DATA/T_target.pt')
        V_data = torch.load('T_' + dataname + '_DATA/V_data.pt')
        V_target = torch.load('T_' + dataname + '_DATA/V_target.pt')
        TE_data = torch.load('T_' + dataname + '_DATA/TE_data.pt')
        TE_target = torch.load('T_' + dataname + '_DATA/TE_target.pt')

        T_data = torch.transpose(T_data, 0, 1)
        T_data = T_data[:100]
        T_data = torch.transpose(T_data, 0, 1)
        print(T_data.size())

        V_data = torch.transpose(V_data, 0, 1)
        V_data = V_data[:100]
        V_data = torch.transpose(V_data, 0, 1)

        TE_data = torch.transpose(TE_data, 0, 1)
        TE_data = TE_data[:100]
        TE_data = torch.transpose(TE_data, 0, 1)

        train_datasets = torch.utils.data.TensorDataset(T_data, T_target)
        val_datasets = torch.utils.data.TensorDataset(V_data, V_target)
        test_datasets = torch.utils.data.TensorDataset(TE_data, TE_target)

        #Train

        autoencoder = LitAutoEncoder(pos_type=pos_bool)
        early_stopping = EarlyStopping('val_loss')
        #print(autoencoder.decoder2.weight, autoencoder.decoder2.bias)

        trainer = pl.Trainer(
            max_epochs=10000,
            callbacks = [early_stopping] #,
            #gpus = 1,
            #accelerator='dp'
        )

        DL_Train = DataLoader(train_datasets, batch_size=8)
        DL_Val = DataLoader(val_datasets, batch_size=1)
        DL_Test = DataLoader(test_datasets)

        trainer.fit(
            autoencoder,
            DL_Train,
            DL_Val
        )
        trainer.test(
            test_dataloaders=DL_Train
        )

experiment('T', True)
experiment('T', False)
experiment('Depth')
experiment('Time')