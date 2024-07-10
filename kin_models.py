import torch
import torch.nn as nn
import warnings
import numpy as np


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(106, 2400),
            nn.ReLU(inplace=True),
            nn.Linear(2400, 2400),
            nn.ReLU(inplace=True),
            nn.Linear(2400, 1200),
            nn.ReLU(inplace=True),
            nn.Linear(1200, 1200),
            nn.ReLU(inplace=True),
            nn.Linear(1200, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 2),
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x
    
class FC_Reg(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super(FC_Reg, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(106, 2400),
            nn.BatchNorm1d(2400),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(2400, 2400),
            nn.BatchNorm1d(2400),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(2400, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1200, 1200),
            nn.BatchNorm1d(1200),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1200, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(600, 600),
            nn.BatchNorm1d(600),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(600, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(300, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(300, 2),
        )
    
    def forward(self, x):
        return self.mlp(x)
    
class eMLP(torch.nn.Module):
    def __init__(self, input_dim=106, output_dim=2, hidden_dim=200, dropout=0.3):
        super(eMLP, self).__init__()

        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),


            torch.nn.Linear(int(hidden_dim), int(hidden_dim / 4)),
            torch.nn.BatchNorm1d(int(hidden_dim / 4)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(int(hidden_dim / 4), int(hidden_dim / 8)),
            torch.nn.BatchNorm1d(int(hidden_dim / 8)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout)
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(int(hidden_dim/8), int(hidden_dim / 4)),
            torch.nn.BatchNorm1d(int(hidden_dim / 4)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),


            torch.nn.Linear(int(hidden_dim / 4), hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
        )

        self.output_network = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_network(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=6, d_model = 128, nhead = 8, dim_feedforward = 256, input_dim = 106, output_dim = 2, dropout = 0.3):
        super(TransformerEncoder, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.dropout = dropout
        self.padding = self.d_model - self.input_dim
        assert self.padding > 0, f"Padding Error: Number bigger or equal then 0 expected but got {self.padding} as padding."

        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,dropout=dropout,batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.output = torch.nn.Sequential(
            torch.nn.Linear(d_model,int(d_model/2)),
            torch.nn.BatchNorm1d(int(d_model/2)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=self.dropout),

            torch.nn.Linear(int(d_model/2),d_model),
            torch.nn.BatchNorm1d(d_model),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=self.dropout),

            torch.nn.Linear(d_model,output_dim)
        )

    def forward(self, kin):
        x = torch.nn.functional.pad(kin, (0,self.padding))
        encoding = self.transformer_encoder(x)
        out = self.output(encoding)
        return out