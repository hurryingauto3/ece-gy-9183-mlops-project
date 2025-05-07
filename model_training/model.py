# model_training/model.py
import torch
import torch.nn as nn
# Add other necessary imports like torch.nn.utils.rnn if any layer uses it directly here

class LSTMTCNRegressor(nn.Module):
    """
    LSTM and TCN based regressor model incorporating county FIPS embeddings.

    Args:
        input_dim (int): Number of features in the weather time series.
        num_fips (int): Total number of unique FIPS codes for the embedding layer.
        fips_embedding_dim (int): Dimension of the FIPS embedding vector.
        hidden_dim (int): Hidden dimension of the LSTM layer.
        lstm_layers (int): Number of layers in the LSTM.
        tcn_channels (list): List of output channels for each TCN Conv1d layer.
        dropout_rate (float): Dropout rate for LSTM and TCN layers.
    """
    def __init__(self, input_dim, num_fips, fips_embedding_dim=16, hidden_dim=64, lstm_layers=1, tcn_channels=[64, 32], dropout_rate=0.1):
        super(LSTMTCNRegressor, self).__init__()

        # Ensure num_fips is at least 1 for the embedding layer dimension calculation
        # An embedding layer of size (0, X) is invalid
        if num_fips <= 0:
             # Handle case where no FIPS codes were found in data
             # You might train a model without FIPS embedding or raise an error
             # For now, let's adjust num_fips minimally or raise
             # Raising is better for job failure notification
             raise ValueError(f"num_fips must be > 0, but got {num_fips}. No FIPS data found?")

        # Embedding layer for FIPS codes
        # Use padding_idx=0 if FIPS ID 0 is reserved for padding or unknown (dataset assigns 0+)
        self.fips_embedding = nn.Embedding(num_fips, fips_embedding_dim)

        # Input to LSTM will be weather_features + fips_embedding_dim
        lstm_input_dim = input_dim + fips_embedding_dim

        # LSTM layer to process the sequence of combined weather and FIPS features
        # dropout > 0 is only applied between layers if num_layers > 1
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout_rate if lstm_layers > 1 else 0)

        # TCN part using 1D Convolutions
        tcn_layers = []
        in_channels = hidden_dim # Input channels = LSTM output size
        for i, out_channels in enumerate(tcn_channels):
             # Conv1D kernel size 3, padding 1 maintains sequence length
             tcn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
             tcn_layers.append(nn.ReLU()) # Activation after convolution
             if dropout_rate > 0: # Apply dropout after activation
                 tcn_layers.append(nn.Dropout(dropout_rate))
             in_channels = out_channels # Output becomes next input

        self.tcn = nn.Sequential(*tcn_layers)

        # Adaptive pooling after TCN to get a fixed-size vector (sequence dimension is pooled)
        self.pooling = nn.AdaptiveAvgPool1d(1) # Output shape: (batch, channels, 1)

        # Final fully connected layer to predict a single value (yield)
        self.fc = nn.Linear(tcn_channels[-1], 1) # Input size = channels from last TCN layer

        self.dropout_rate = dropout_rate # Store dropout rate (useful for MC Dropout if needed elsewhere)

    def forward(self, x_weather, fips_ids):
        """
        Forward pass of the model.

        Args:
            x_weather (torch.Tensor): Padded weather time series data (batch, time_steps, weather_features).
            fips_ids (torch.Tensor): Batch of FIPS integer IDs (batch,).

        Returns:
            torch.Tensor: Predicted yield for each sample in the batch (batch,).
        """
        # Get county embeddings from FIPS IDs
        fips_emb = self.fips_embedding(fips_ids) # Shape: (batch, fips_embedding_dim)

        # Repeat the FIPS embedding along the time dimension to match the weather sequence length
        # unsqueeze(1) adds a dimension for time (current size 1), repeat expands it
        fips_emb_expanded = fips_emb.unsqueeze(1).repeat(1, x_weather.size(1), 1) # Shape: (batch, time_steps, fips_embedding_dim)

        # Concatenate weather features and county embeddings along the feature dimension (-1)
        x_combined = torch.cat([x_weather, fips_emb_expanded], dim=-1) # Shape: (batch, time_steps, weather_features + fips_embedding_dim)

        # Pass the combined data through the LSTM
        # out shape: (batch, time_steps, hidden_dim)
        # h_n, c_n are the hidden and cell states for the last time step (ignored here)
        out, _ = self.lstm(x_combined)

        # Permute the output shape for Conv1D (needs channels dimension before time dimension)
        # From (batch, time_steps, hidden_dim) to (batch, hidden_dim, time_steps)
        out = out.permute(0, 2, 1)

        # Pass through the TCN layers
        out = self.tcn(out) # Shape: (batch, tcn_channels[-1], time_steps)

        # Apply adaptive pooling to get a single vector representation per sample
        out = self.pooling(out) # Shape: (batch, tcn_channels[-1], 1)

        # Remove the last dimension (the single pooling dimension)
        out = out.squeeze(-1) # Shape: (batch, tcn_channels[-1])

        # Pass through the final fully connected layer to get the predicted yield
        out = self.fc(out) # Shape: (batch, 1)

        # Squeeze the last dimension to get a 1D tensor of predictions (batch,)
        return out.squeeze(-1)