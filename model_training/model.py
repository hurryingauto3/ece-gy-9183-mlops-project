# model.py
import torch
import torch.nn as nn

class LSTMTCNRegressor(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_fips:  int,
                 fips_embedding_dim: int = 16,
                 hidden_dim:         int = 64,
                 lstm_layers:        int = 1,
                 tcn_channels:       list = [64, 32],
                 dropout_rate:     float = 0.1):
        super().__init__()

        self.fips_embedding = nn.Embedding(num_fips, fips_embedding_dim)
        lstm_in = input_dim + fips_embedding_dim
        self.lstm = nn.LSTM(lstm_in, hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout_rate if lstm_layers>1 else 0)

        layers, in_ch = [], hidden_dim
        for out_ch in tcn_channels:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                       nn.ReLU(),
                       nn.Dropout(dropout_rate)]
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(tcn_channels[-1], 1)

    def forward(self, x_weather, fips_ids):
        # x_weather: (B, T, F)
        emb = self.fips_embedding(fips_ids)            # (B, E)
        emb = emb.unsqueeze(1).repeat(1, x_weather.size(1), 1)
        x   = torch.cat([x_weather, emb], dim=-1)      # (B, T, F+E)
        out, _ = self.lstm(x)                          # (B, T, H)
        out = out.permute(0,2,1)                       # (B, H, T)
        out = self.tcn(out)                            # (B, C, T)
        out = self.pool(out).squeeze(-1)               # (B, C)
        return self.fc(out).squeeze(-1)                # (B,)
