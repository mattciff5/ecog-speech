import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from scipy import signal
from scipy.io import wavfile
from scipy.stats import pearsonr, zscore
from mne_bids import BIDSPath
from functools import partial
from nilearn.plotting import plot_markers
import torch
from torch import nn
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import random



class LearnableTau(nn.Module):
    def __init__(self, init_tau=0.05):
        super().__init__()
        self.log_tau = nn.Parameter(torch.log(torch.tensor(init_tau)))

    def forward(self):
        return self.log_tau.exp() 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class AttentiveStim2BrainNet(nn.Module):
    def __init__(self, input_dim=512, output_channels=235, time_in=250, time_out=282, d_model=256, nhead=2, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=time_in)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        # self.norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.temporal_upsample = nn.Upsample(size=time_out, mode='linear', align_corners=True)

        self.mh_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.output_proj = nn.Linear(d_model, output_channels)

    def forward(self, x):   # x: (batch, time_in, input_dim=512)
        x = self.input_proj(x)                           # (batch, time_in, d_model)
        x = self.pos_enc(x)                             
        x_encoded = self.encoder(x)                             
        x_target = self.temporal_upsample(x_encoded.permute(0, 2, 1)).permute(0, 2, 1)   # (batch, time_out, d_model)
        x_attn, attn_weights = self.mh_attention(query=x_target, key=x_encoded, value=x_encoded)  # (batch, time_out, d_model)    
        # x = self.norm(x_attn)
        x = self.output_proj(x_attn)                    # (batch, time_out, output_channels)
        return x.permute(0, 2, 1), attn_weights                       # (batch, output_channels, time_out)
    

class SoftMappingGRUSeq(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, time_out=282, output_channels=235):
        super().__init__()
        
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.time_queries = nn.Parameter(torch.randn(time_out, hidden_dim * 2))  # (time_out, d)
        self.attn_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)  # to match key dim
        self.output_proj = nn.Linear(hidden_dim * 2, output_channels)
        self.norm = torch.nn.LayerNorm(512)
        # self.contrastive_proj = ContrastiveProjection(input_dim=output_channels * time_out)

    def forward(self, x):  # x: (batch, time_in, 512)
        # x = self.norm(x)
        h, _ = self.encoder_gru(x)  # (batch, time_in, d*2)

        # Compute attention: time_out queries → weighted sum over time_in
        q = self.time_queries.unsqueeze(0).expand(x.size(0), -1, -1)  # (batch, time_out, d*2)
        k = self.attn_proj(h)  # (batch, time_in, d*2)

        attn_weights = torch.matmul(q, k.transpose(1, 2))  # (batch, time_out, time_in)
        attn_weights = F.softmax(attn_weights, dim=-1)
        context = torch.bmm(attn_weights, h)  # (batch, time_out, d*2)
        out = self.output_proj(context)       # (batch, time_out, channels)
        return out.permute(0, 2, 1), attn_weights
    

class Audio2BrainCNN(nn.Module):
    def __init__(self, input_time=110, dim_feature=512, output_time=128, output_channels=235):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(dim_feature, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, output_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.out_proj = nn.Linear(input_time, output_time)
        self.output_time = output_time

    def forward(self, x):  # x: (batch, 110, 512)
        
        x = x.permute(0, 2, 1)        # (batch, 512, 110)
        x_conv = self.conv(x)              # (batch, 235, 110)
        x_conv = self.out_proj(x_conv)  # (batch, 235, output_time)
        return x_conv, x
    


class LinearMLPModel(nn.Module):
    def __init__(self, feature_dim=512, out_channels=235, timepoint_out=128, timepoint_in=110, hidden_feat=256):
        super().__init__()
        self.linear = nn.Linear(feature_dim, out_channels)
        # self.linear = nn.Sequential(
        #     nn.Linear(feature_dim, hidden_feat),
        #     nn.ReLU(),
        #     nn.Linear(hidden_feat, out_channels)
        # )
        self.proj_time = nn.Linear(timepoint_in, timepoint_out)

    def forward(self, x):
        # x: (batch, timepoint_in, feature)
        x = self.linear(x)  # (batch, timepoint_in, out_channels)
        x = x.permute(0, 2, 1) 
        x = self.proj_time(x)    # (batch, out_channels, timepoint_out)
        return x
    