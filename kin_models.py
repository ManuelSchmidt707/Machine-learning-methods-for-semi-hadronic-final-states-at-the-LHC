import torch
import torch.nn as nn
import warnings
import numpy as np

class FC(nn.Module):
    """
    A fully connected neural network for FC from "Uncovering doubly charged scalars with dominant three-body decays using machine learning"
    """
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

class FC_Reg(nn.Module):
    """
    The FC Network with added regularization layers.
    """
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

class eMLP(nn.Module):
    """
    An encoder-decoder Multi-Layer Perceptron for classification.

    The architecture consists of an encoder and decoder with dropout for regularization.
    We use all 106 kinematic variables for the classification task.

    """
    def __init__(self, input_dim=106, output_dim=2, hidden_dim=2000, dropout=0.3):
        super(eMLP, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.BatchNorm1d(hidden_dim // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 8, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.output_network = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_network(x)
        return x

class TransformerEncoder(nn.Module):
    """
    A Transformer encoder model for sequence classification.

    The architecture consists of transformer encoder layers followed by fully connected layers for output.

    """
    def __init__(self, num_layers=6, d_model=128, nhead=8, dim_feedforward=256, input_dim=106, output_dim=2, dropout=0.3):
        super(TransformerEncoder, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.dropout = dropout
        self.padding = self.d_model - self.input_dim
        assert self.padding > 0, f"Padding Error: Padding should be greater than 0 but got {self.padding}."

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.output = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),

            nn.Linear(d_model // 2, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),

            nn.Linear(d_model, output_dim)
        )

    def forward(self, kin):

        x = nn.functional.pad(kin, (0, self.padding))
        encoding = self.transformer_encoder(x)
        out = self.output(encoding)
        return out

