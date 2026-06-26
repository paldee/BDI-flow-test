import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        # Input shape: (Batch, 1, 10001)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        # Expected output shape: (Batch, 256, 313)

    def forward(self, x):
        return self.conv(x)

class NMRTransformerModel(nn.Module):
    def __init__(self, num_classes=1328, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(NMRTransformerModel, self).__init__()
        
        self.cnn = CNNFeatureExtractor()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, 10001)
        cnn_out = self.cnn(x) # (B, d_model, SeqLen) e.g., (B, 256, 313)
        
        # Transformer expects (SeqLen, B, d_model)
        seq_out = cnn_out.permute(2, 0, 1) # (SeqLen, B, d_model)
        seq_out = self.pos_encoder(seq_out)
        
        tx_out = self.transformer_encoder(seq_out) # (SeqLen, B, d_model)
        
        # Global Average Pooling across the sequence
        # Permute back to (B, SeqLen, d_model) then take mean over SeqLen
        tx_out_batch_first = tx_out.permute(1, 0, 2)
        pooled = tx_out_batch_first.mean(dim=1) # (B, d_model)
        
        logits = self.classifier(pooled) # (B, num_classes)
        return logits
