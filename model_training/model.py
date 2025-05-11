import torch
import torch.nn as nn
import structlog # Added for logger

logger = structlog.get_logger(__name__) # Added for logger

class LSTMTCNHistogramPredictor(nn.Module):
    def __init__(self,
                 input_dim: int,          # Weather features dimension
                 num_fips:  int,          # Number of unique FIPS codes for embedding
                 num_crops: int,          # Number of unique crop types for embedding
                 num_bins: int,           # Number of bins for the output histogram
                 fips_embedding_dim: int = 16,
                 crop_embedding_dim: int = 8,
                 hidden_dim:         int = 64,
                 lstm_layers:        int = 1,
                 tcn_channels:       list = [64, 32],
                 dropout_rate:     float = 0.1):
        super().__init__()

        if num_fips <= 0:
            raise ValueError(f"num_fips must be > 0, got {num_fips}")
        if num_crops <= 0:
            raise ValueError(f"num_crops must be > 0, got {num_crops}")
        if num_bins <= 0:
            raise ValueError(f"num_bins must be > 0, got {num_bins}")

        self.fips_embedding = nn.Embedding(num_fips, fips_embedding_dim)
        self.crop_embedding = nn.Embedding(num_crops, crop_embedding_dim)
        
        # LSTM input dimension: weather features + FIPS embedding + crop embedding
        lstm_in = input_dim + fips_embedding_dim + crop_embedding_dim
        
        self.lstm = nn.LSTM(lstm_in, hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout_rate if lstm_layers > 1 else 0)

        layers, in_ch = [], hidden_dim
        for out_ch in tcn_channels:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                       nn.ReLU(),
                       nn.Dropout(dropout_rate)]
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Final layer outputs probabilities for each bin
        self.fc   = nn.Linear(tcn_channels[-1], num_bins)
        # Softmax will be applied after this to get probabilities if needed, or use CrossEntropyLoss during training

    def forward(self, x_weather, fips_ids, crop_ids):
        # x_weather: (B, T, F_weather)
        # fips_ids: (B,)
        # crop_ids: (B,)
        
        fips_emb = self.fips_embedding(fips_ids)            # (B, E_fips)
        crop_emb = self.crop_embedding(crop_ids)            # (B, E_crop)
        
        # Expand embeddings to match time dimension T of x_weather
        fips_emb_expanded = fips_emb.unsqueeze(1).repeat(1, x_weather.size(1), 1) # (B, T, E_fips)
        crop_emb_expanded = crop_emb.unsqueeze(1).repeat(1, x_weather.size(1), 1) # (B, T, E_crop)
        
        # Concatenate weather features and embeddings
        x = torch.cat([x_weather, fips_emb_expanded, crop_emb_expanded], dim=-1)      # (B, T, F_weather + E_fips + E_crop)
        
        out, _ = self.lstm(x)                          # (B, T, H)
        out = out.permute(0,2,1)                       # (B, H, T)
        out = self.tcn(out)                            # (B, C, T) # C is last TCN channel output
        out = self.pool(out).squeeze(-1)               # (B, C)
        
        # Output raw scores (logits) for each bin. Softmax can be applied outside if needed.
        out = self.fc(out)                             # (B, num_bins)
        return out


class DummyLSTMTCNHistogramPredictor(nn.Module):
    def __init__(self,
                 input_dim: int,          # Weather features dimension (not strictly used by dummy)
                 num_fips:  int,          # Number of unique FIPS codes (for signature match)
                 num_crops: int,          # Number of unique crop types (for signature match)
                 num_bins: int,           # Number of bins for the output histogram
                 fips_embedding_dim: int = 16, # Not used
                 crop_embedding_dim: int = 8,  # Not used
                 hidden_dim:         int = 64, # Not used
                 lstm_layers:        int = 1,  # Not used
                 tcn_channels:       list = [64, 32], # Not used
                 dropout_rate:     float = 0.1): # Not used
        super().__init__()
        if num_bins <= 0:
            raise ValueError(f"num_bins must be > 0, got {num_bins}")
        self.num_bins = num_bins
        # No actual layers needed, just to satisfy potential instantiation if not loading a saved one.
        self.dummy_param = nn.Parameter(torch.empty(0)) # To make it a valid nn.Module with parameters

        logger.info(f"DummyLSTMTCNHistogramPredictor initialized with num_bins: {self.num_bins}")

    def forward(self, x_weather, fips_ids, crop_ids):
        # x_weather: (B, T, F_weather)
        # fips_ids: (B,)
        # crop_ids: (B,)
        
        batch_size = x_weather.size(0)
        # Output random logits for each bin
        # Ensure it's on the same device as input if inputs were on GPU (though for CPU dummy this is fine)
        dummy_logits = torch.randn(batch_size, self.num_bins, device=x_weather.device)
        # logger.debug(f"Dummy model returning logits of shape: {dummy_logits.shape} for batch_size {batch_size}")
        return dummy_logits
