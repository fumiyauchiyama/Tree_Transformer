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

'''
ntokens = 100  # the size of vocabulary
emsize = 768  # embedding dimension
nhid = emsize * 2  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 1  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # the number of heads in the multiheadattention models
dropout = 0.1  # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout)
'''

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

class LitAutoEncoder(pl.LightningModule):

    def __init__(self, ninp=768, nhead=8, nhid=768*2, nlayers=1, dropout=0.1):
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

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        x = x.transpose(1, 0)

        if self.src_mask is None or self.src_mask.size(0) != x.size(0):
            device = x.device
            mask = self._generate_square_subsequent_mask(x.size(0)).to(device)
            self.src_mask = mask

        #x = self.pos_encoder(x)
        embedding = self.transformer_encoder(x, self.src_mask)
        #embedding = self.transformer_encoder(x)
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
        #print(loss)
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


#Dataload

for i in range(5):
    torch.manual_seed(i)

    T_data = torch.load('T_Time_DATA/T_data.pt')
    T_target = torch.load('T_Time_DATA/T_target.pt')
    V_data = torch.load('T_Time_DATA/V_data.pt')
    V_target = torch.load('T_Time_DATA/V_target.pt')
    TE_data = torch.load('T_Time_DATA/TE_data.pt')
    TE_target = torch.load('T_Time_DATA/TE_target.pt')

    T_data = torch.transpose(T_data, 0, 1)
    T_data = T_data[:100]
    T_data = torch.transpose(T_data, 0, 1)

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

    autoencoder = LitAutoEncoder()
    early_stopping = EarlyStopping('val_loss')
    #print(autoencoder.decoder2.weight, autoencoder.decoder2.bias)

    trainer = pl.Trainer(
        max_epochs=10000,
        callbacks = [early_stopping]
    )
    trainer.fit(
        autoencoder,
        DataLoader(train_datasets, batch_size=8),
        DataLoader(val_datasets)
    )
    trainer.test(
        test_dataloaders=DataLoader(test_datasets)
    )



