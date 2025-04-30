# %% [markdown]
# ## Imports
# Import necessary libraries: torch, pandas, json, os, pathlib, numpy, matplotlib.
# torch.nn is for neural network layers, torch.utils.data for data handling.

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.utils.rnn  # Needed for pad_sequence

import pandas as pd
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tqdm  # For progress bars
from sklearn.metrics import mean_squared_error, mean_absolute_error

# %% [markdown]
# ## Dataset Class
# Defines a custom Dataset to load weather time series and annual yield data.
# It handles organizing data by FIPS code and year, filtering to the April-October growing season,
# and mapping FIPS codes to unique integer IDs required for embedding layers.


# %%
class CropYieldDataset(Dataset):
    """
    Loads crop yield data along with corresponding weather time series for specific counties and years.

    Organizes data from a specified directory structure:
    data_lake_dir/
    ├── FIPS_CODE_1/
    │   ├── crop_name.json (contains yearly yield data for this FIPS)
    │   ├── YEAR_1/
    │   │   └── WeatherTimeSeriesYEAR_1.csv (daily weather data)
    │   ├── YEAR_2/
    │   │   └── WeatherTimeSeriesYEAR_2.csv
    │   └── ...
    └── FIPS_CODE_2/
        └── ...
    """

    def __init__(
        self,
        data_lake_dir="../data/data_lake_organized",
        crop_name="corn",
        transform=None,
    ):
        # Stores samples as tuples: (weather_tensor, yield_target, fips_id)
        self.samples = []
        self.transform = transform  # Optional transformations on weather data
        self.crop_name = crop_name.lower()

        # Dictionaries to map FIPS string codes to integer IDs and vice-versa
        self.fips_to_id = {}
        self.id_to_fips = {}
        self._next_fips_id = 0  # Counter for assigning unique integer IDs

        data_lake_path = Path(data_lake_dir)
        if not data_lake_path.exists():
            raise FileNotFoundError(
                f"Data lake directory not found at: {data_lake_dir}"
            )

        # Iterate through each FIPS code folder in the data lake directory
        fips_folders = [f for f in data_lake_path.iterdir() if f.is_dir()]

        if not fips_folders:
            print(f"Warning: No FIPS folders found in {data_lake_dir}.")

        for fips_folder in tqdm.tqdm(fips_folders, desc="Loading Data"):
            fips_code = fips_folder.name  # Folder name is the FIPS code (string)

            # Assign a unique integer ID if this FIPS code hasn't been seen before
            if fips_code not in self.fips_to_id:
                self.fips_to_id[fips_code] = self._next_fips_id
                self.id_to_fips[self._next_fips_id] = fips_code
                self._next_fips_id += 1

            fips_id = self.fips_to_id[fips_code]

            # Check if the crop's yield JSON file exists for this FIPS
            crop_json_path = fips_folder / f"{self.crop_name}.json"
            if not crop_json_path.exists():
                # print(f"Skipping FIPS {fips_code}: No {self.crop_name}.json found.") # Optional detailed log
                continue  # Skip this FIPS if no yield data for the target crop

            # Load the yield data for this crop and FIPS
            try:
                with open(crop_json_path, "r") as f:
                    yield_data = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not decode JSON from {crop_json_path}. Skipping."
                )
                continue  # Skip if JSON is invalid
            except Exception as e:
                print(f"Warning: Error reading {crop_json_path}: {e}. Skipping.")
                continue  # Skip on other file errors

            # Iterate through year folders within the FIPS folder
            year_folders = [y for y in fips_folder.iterdir() if y.is_dir()]

            for year_folder in year_folders:
                year_str = year_folder.name  # Year is a string from folder name

                # Check if yield data exists for this specific year
                if year_str not in yield_data or "yield" not in yield_data[year_str]:
                    # print(f"Skipping FIPS {fips_code}, Year {year_str}: No yield data found.") # Optional detailed log
                    continue  # Skip this year if no yield data

                # Check if the weather CSV exists for this year and FIPS
                weather_csv = year_folder / f"WeatherTimeSeries{year_str}.csv"
                if not weather_csv.exists():
                    # print(f"Skipping FIPS {fips_code}, Year {year_str}: Weather CSV missing.") # Optional detailed log
                    continue  # Skip this year if weather data is missing

                # Load the weather data
                try:
                    df = pd.read_csv(weather_csv)
                except pd.errors.EmptyDataError:
                    print(
                        f"Warning: Weather CSV empty for FIPS {fips_code}, Year {year_str}. Skipping."
                    )
                    continue
                except Exception as e:
                    print(f"Warning: Error reading {weather_csv}: {e}. Skipping.")
                    continue

                # Filter weather data to the growing season (April to October)
                df_season = df[
                    (df["Month"] >= 4) & (df["Month"] <= 10)
                ].copy()  # Use .copy() to avoid SettingWithCopyWarning

                if df_season.empty:
                    # print(f"Skipping FIPS {fips_code}, Year {year_str}: No weather data found in Apr-Oct.") # Optional detailed log
                    continue  # Skip if no weather data in the target range

                # Drop non-weather columns like Year, Month, Day
                cols_to_drop = ["Year", "Month", "Day"]
                existing_cols_to_drop = [
                    col for col in cols_to_drop if col in df_season.columns
                ]
                df_season = df_season.drop(
                    columns=existing_cols_to_drop, errors="ignore"
                )

                # Ensure all weather columns have numeric types
                for col in df_season.columns:
                    # Attempt conversion, non-numeric will become NaN
                    df_season[col] = pd.to_numeric(df_season[col], errors="coerce")

                # Drop any columns that became all NaN after conversion, or have very few non-NaN values
                df_season.dropna(axis=1, how="all", inplace=True)
                # Optional: Drop columns with too many NaNs (e.g., more than 10% missing per year)
                # df_season.dropna(axis=1, thresh=int(len(df_season)*0.9), inplace=True)

                # Drop rows with any NaN values (missing days within the season)
                # Or impute NaNs if appropriate (e.g., forward fill, mean)
                initial_rows = len(df_season)
                df_season.dropna(axis=0, how="any", inplace=True)
                if len(df_season) < initial_rows:
                    print(
                        f"Warning: Dropped {initial_rows - len(df_season)} rows with NaN weather data for FIPS {fips_code}, Year {year_str}."
                    )
                    if df_season.empty:
                        print(
                            f"Skipping FIPS {fips_code}, Year {year_str}: No valid weather data rows left."
                        )
                        continue

                # Convert the cleaned weather data to a PyTorch tensor
                weather_tensor = torch.tensor(df_season.values, dtype=torch.float32)

                # Get the yield target and convert to tensor
                yield_target = torch.tensor(
                    yield_data[year_str]["yield"], dtype=torch.float32
                )

                # Add the sample to our list
                self.samples.append((weather_tensor, yield_target, fips_id))

        print(f"\nFinished loading data.")
        print(
            f"Loaded {len(self.samples)} samples for crop '{self.crop_name}'. Found {len(self.fips_to_id)} unique FIPS codes."
        )

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Retrieves a single sample by index."""
        # Sample is (weather_tensor, yield_target, fips_id)
        x, y, fips = self.samples[idx]
        # Apply transform only to the weather tensor if specified
        if self.transform:
            x = self.transform(x)
        # Return the sample components
        return x, y, fips

    def get_fips_mapping(self):
        """Returns the dictionaries mapping FIPS codes to IDs and vice-versa."""
        return self.fips_to_id, self.id_to_fips

    def get_num_fips(self):
        """Returns the number of unique FIPS codes found."""
        return len(self.fips_to_id)


# %% [markdown]
# ## Collate Function
# Custom `collate_fn` for the DataLoader. This is necessary because weather sequences
# have varying lengths and need to be padded to create batches. It also correctly
# stacks the yield targets and FIPS IDs.


# %%
def collate_fn(batch):
    """
    Collates a batch of samples. Pads weather sequences and stacks targets and FIPS IDs.

    Args:
        batch (list): A list of samples, where each sample is a tuple
                      (weather_tensor, yield_target, fips_id).

    Returns:
        tuple: Padded weather tensor batch, stacked yield target batch, stacked FIPS ID batch.
    """
    # Separate the elements of the batch
    # batch is a list of tuples: [(weather_tensor_1, yield_target_1, fips_id_1), ...]
    xs = [item[0] for item in batch]  # List of weather_tensors (varying lengths)
    ys = [item[1] for item in batch]  # List of yield_targets (scalar tensors)
    fips_ids = [item[2] for item in batch]  # List of fips_ids (Python integers)

    # 1. Pad the sequences (weather tensors)
    # pad_sequence handles the variable lengths by adding padding_value
    # batch_first=True arranges the output as (batch_size, max_seq_len, num_features)
    xs_padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0.0)

    # 2. Stack the targets
    # yield targets are scalars, stack them into a batch tensor (batch_size,)
    ys_stacked = torch.stack(ys)

    # 3. Convert FIPS IDs (Python integers) to tensors and stack them
    # Each fips_id is an integer index, convert to torch.long (required for embeddings)
    fips_ids_tensors = [torch.tensor(fips_id, dtype=torch.long) for fips_id in fips_ids]
    fips_ids_stacked = torch.stack(
        fips_ids_tensors
    )  # Stack into a batch tensor (batch_size,)

    # Return the batch as a tuple of tensors
    return xs_padded, ys_stacked, fips_ids_stacked


# %% [markdown]
# ## Model Definition
# Defines the `LSTMTCNRegressor` neural network model.
# This model uses an Embedding layer for county FIPS IDs, an LSTM for processing weather sequences,
# a TCN for capturing local temporal features, and a final Linear layer to predict yield.


# %%
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

    def __init__(
        self,
        input_dim,
        num_fips,
        fips_embedding_dim=16,
        hidden_dim=64,
        lstm_layers=1,
        tcn_channels=[64, 32],
        dropout_rate=0.1,
    ):
        super(LSTMTCNRegressor, self).__init__()

        # Embedding layer for FIPS codes
        # Maps each integer FIPS ID to a dense vector representation
        self.fips_embedding = nn.Embedding(num_fips, fips_embedding_dim)

        # The input to the LSTM will be the concatenation of weather features and the FIPS embedding
        lstm_input_dim = input_dim + fips_embedding_dim

        # LSTM layer to process the sequence of combined weather and FIPS features
        # batch_first=True means input/output shape is (batch_size, seq_len, features)
        # dropout applies dropout to the output of each LSTM layer except the last
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
        )

        # TCN part using 1D Convolutions
        # Processes the sequence of features output by the LSTM
        tcn_layers = []
        in_channels = (
            hidden_dim  # Input channels to TCN is the output dimension of the LSTM
        )
        for i, out_channels in enumerate(tcn_channels):
            tcn_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            )  # padding=1 maintains sequence length
            tcn_layers.append(nn.ReLU())  # Non-linear activation
            if dropout_rate > 0:
                tcn_layers.append(nn.Dropout(dropout_rate))  # Dropout after activation
            in_channels = out_channels  # Output channels of current layer become input for the next

        self.tcn = nn.Sequential(*tcn_layers)

        # Adaptive pooling after TCN to get a fixed-size vector regardless of sequence length
        # AdaptiveAvgPool1d(1) takes the average across the sequence dimension
        self.pooling = nn.AdaptiveAvgPool1d(1)  # output shape: (batch, channels, 1)

        # Final fully connected layer to predict the yield (single scalar output)
        self.fc = nn.Linear(
            tcn_channels[-1], 1
        )  # Input dimension is the output channels of the last TCN layer

        self.dropout_rate = dropout_rate  # Store dropout rate for potential use in forward (e.g., for MC Dropout)

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
        fips_emb = self.fips_embedding(fips_ids)  # Shape: (batch, fips_embedding_dim)

        # Repeat the FIPS embedding along the time dimension to match the weather sequence length
        # Unsqueeze adds a dimension for sequence length (which is 1 here)
        # Repeat copies the embedding along the time dimension `x_weather.size(1)` times
        fips_emb_expanded = fips_emb.unsqueeze(1).repeat(
            1, x_weather.size(1), 1
        )  # Shape: (batch, time_steps, fips_embedding_dim)

        # Concatenate weather features and county embeddings along the last dimension (features dimension)
        x_combined = torch.cat(
            [x_weather, fips_emb_expanded], dim=-1
        )  # Shape: (batch, time_steps, weather_features + fips_embedding_dim)

        # Pass the combined data through the LSTM
        # out shape: (batch, time_steps, hidden_dim)
        # h_n, c_n are the hidden and cell states for the last time step (ignored here)
        out, (h_n, c_n) = self.lstm(x_combined)

        # Permute the output shape for Conv1D (needs channels dimension before time dimension)
        # From (batch, time_steps, hidden_dim) to (batch, hidden_dim, time_steps)
        out = out.permute(0, 2, 1)

        # Pass through the TCN layers
        out = self.tcn(
            out
        )  # Shape: (batch, tcn_channels[-1], time_steps) (padding keeps length)

        # Apply adaptive pooling to get a single vector representation per sample
        out = self.pooling(out)  # Shape: (batch, tcn_channels[-1], 1)

        # Remove the last dimension (the single pooling dimension)
        out = out.squeeze(-1)  # Shape: (batch, tcn_channels[-1])

        # Pass through the final fully connected layer to get the predicted yield
        out = self.fc(out)  # Shape: (batch, 1)

        # Squeeze the last dimension to get a 1D tensor of predictions (batch,)
        return out.squeeze(-1)


# %% [markdown]
# ## Data Loading and Splitting
# Instantiate the dataset, get the number of unique FIPS codes, and split the dataset
# into training and validation sets using `random_split`. Create DataLoaders for
# each set, using the custom `collate_fn`.


# %%
def get_dataloaders(dataset, train_ratio=0.8, batch_size=32):
    """
    Splits the dataset into training and validation sets and returns DataLoaders.

    Args:
        dataset (Dataset): The CropYieldDataset instance.
        train_ratio (float): The proportion of the dataset to use for training.
        batch_size (int): The batch size for the DataLoaders.

    Returns:
        tuple: (train_loader, val_loader) - DataLoaders for training and validation.
    Raises:
        ValueError: If the dataset is too small for a train/validation split.
    """
    total_len = len(dataset)
    if total_len == 0:
        raise ValueError("Dataset is empty. Cannot create dataloaders.")

    train_len = int(total_len * train_ratio)
    val_len = total_len - train_len

    # Ensure both train and validation sets have at least one sample
    if train_len == 0 or val_len == 0:
        raise ValueError(
            f"Dataset size ({total_len}) is too small for train/validation split with ratio {train_ratio}. Need at least 2 samples."
        )

    # Split the dataset randomly
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    # Create DataLoaders using the custom collate function
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    print(
        f"\nDataset split into {train_len} training samples and {val_len} validation samples."
    )
    print(f"Training DataLoader created with {len(train_loader)} batches.")
    print(f"Validation DataLoader created with {len(val_loader)} batches.")

    return train_loader, val_loader


# %% [markdown]
# ## Training Function
# Defines the `train_model` function to train the neural network.
# It includes moving data and model to device (GPU/CPU), defining loss and optimizer,
# and iterating through epochs with training and validation steps.


# %%
def train_model(model, train_loader, val_loader, num_epochs=2, lr=1e-3):
    """
    Trains the provided model using the training and validation dataloaders.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate for the Adam optimizer.

    Returns:
        torch.nn.Module: The trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Move model to the selected device

    # Mean Squared Error Loss for regression
    criterion = nn.MSELoss()
    # Adam optimizer for updating model weights
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float(
        "inf"
    )  # Track best validation loss for potential early stopping/saving

    # Loop through epochs
    for epoch in tqdm.tqdm(range(num_epochs), desc="Training Progress"):
        model.train()  # Set model to training mode (enables dropout etc.)
        epoch_loss = 0  # Accumulate loss for the current epoch

        # Iterate over training batches
        for x_batch, y_batch, fips_batch in train_loader:
            # Move batch data to the selected device
            x_batch, y_batch, fips_batch = (
                x_batch.to(device),
                y_batch.to(device),
                fips_batch.to(device),
            )

            # Zero the gradients from the previous iteration
            optimizer.zero_grad()

            # Perform forward pass: get model predictions
            y_pred = model(x_batch, fips_batch)  # Pass weather sequences and FIPS IDs

            # Calculate the loss
            loss = criterion(y_pred, y_batch)

            # Perform backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Perform a single optimization step: update model parameters
            optimizer.step()

            # Accumulate the loss
            epoch_loss += loss.item()

        # --- Validation Step ---
        model.eval()  # Set model to evaluation mode (disables dropout etc.)
        val_loss = 0
        # Disable gradient calculation for validation
        with torch.no_grad():
            for x_val, y_val, fips_val in val_loader:
                x_val, y_val, fips_val = (
                    x_val.to(device),
                    y_val.to(device),
                    fips_val.to(device),
                )
                y_pred = model(x_val, fips_val)  # Get predictions
                val_loss += criterion(
                    y_pred, y_val
                ).item()  # Accumulate validation loss

        # Calculate average losses for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Print epoch statistics using tqdm.write to avoid interfering with the progress bar
        tqdm.tqdm.write(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss} - Val Loss: {avg_val_loss}"
        )
        # Update the progress bar description with the current validation loss
        print(
            f"Training Progress (Epoch {epoch+1} Val Loss: {avg_val_loss})"
        )

        # Optional: Save the best model based on validation loss
        # if avg_val_loss < best_val_loss:
        #      best_val_loss = avg_val_loss
        #      torch.save(model.state_dict(), 'best_model.pth') # Save model state dictionary

    print("\nTraining finished.")
    return model  # Return the fully trained model


# %% [markdown]
# ## Evaluation Function
# Defines the `evaluate_model` function to calculate performance metrics (RMSE, MAE)
# on the validation set after training.


# %%
def evaluate_model(model, val_loader):
    """
    Evaluates the model's performance on the validation dataset.

    Args:
        model (torch.nn.Module): The trained model.
        val_loader (DataLoader): DataLoader for the validation data.

    Returns:
        tuple: (rmse, mae) - Root Mean Squared Error and Mean Absolute Error.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set model to evaluation mode (dropout off)

    y_true = []  # List to store true yield values
    y_pred = []  # List to store predicted yield values

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for x_val, y_val, fips_val in val_loader:  # Unpack FIPS IDs
            x_val, y_val, fips_val = (
                x_val.to(device),
                y_val.to(device),
                fips_val.to(device),
            )
            outputs = model(x_val, fips_val)  # Get predictions
            y_true.extend(y_val.cpu().numpy())  # Collect true values
            y_pred.extend(outputs.cpu().numpy())  # Collect predicted values

    # Calculate RMSE and MAE using scikit-learn
    rmse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\n--- Evaluation Results ---")
    print(f"Final Validation RMSE: {rmse}")
    print(f"Final Validation MAE:  {mae}")
    print(f"--------------------------")

    return rmse, mae


# %% [markdown]
# ## Training Execution
# Instantiate the dataset, prepare dataloaders, determine model input dimensions,
# instantiate the model, and start the training process.

# %%
# --- Instantiate the Dataset ---
# Set the path to your data lake root and the target crop name
DATA_LAKE_ROOT = "data/data_lake_organized"
CROP_NAME = "corn"  # Example crop

dataset = None
try:
    dataset = CropYieldDataset(data_lake_dir=DATA_LAKE_ROOT, crop_name=CROP_NAME)
except FileNotFoundError as e:
    print(f"Fatal Error: {e}")
    print("Please ensure DATA_LAKE_ROOT path is correct and data exists.")

# --- Prepare DataLoaders ---
train_loader, val_loader = None, None
if dataset is not None and len(dataset) > 1:  # Need at least 2 samples to split
    try:
        # Get county mapping and number of unique counties from the dataset
        fips_to_id_mapping, id_to_fips_mapping = dataset.get_fips_mapping()
        num_fips = dataset.get_num_fips()
        print(f"Number of unique FIPS codes found: {num_fips}")

        # Get dataloaders for training and validation
        train_loader, val_loader = get_dataloaders(
            dataset, train_ratio=0.8, batch_size=32
        )

    except ValueError as e:
        print(f"Error creating dataloaders: {e}")
        print("Dataset may be too small for train/validation split after filtering.")
        # train_loader, val_loader remain None

# --- Model Initialization and Training ---
trained_model = None
if train_loader is not None and val_loader is not None:
    # Detect input_dim (number of weather features) from the first sample
    # Accessing dataset[0] is safe here because we checked len(dataset) > 1
    sample_weather, _, _ = dataset[0]
    input_dim = sample_weather.shape[-1]
    print(f"\nDetected input weather dimension: {input_dim}")

    # Instantiate the model
    # Choose an appropriate fips_embedding_dim (e.g., 16, 32, 64)
    model = LSTMTCNRegressor(
        input_dim=input_dim,
        num_fips=num_fips,
        fips_embedding_dim=16,  # Hyperparameter
        hidden_dim=64,  # Hyperparameter
        lstm_layers=1,  # Hyperparameter
        tcn_channels=[64, 32],  # Hyperparameter
        dropout_rate=0.1,  # Hyperparameter
    )

    print(f"\nModel architecture:\n{model}")

    # Train the model
    # Setting epochs low (e.g., 2) for a quick test run
    # Use a higher number (e.g., 30+) for actual training
    trained_model = train_model(
        model, train_loader, val_loader, num_epochs=20
    )  # Reduced epochs for faster demo

    # Evaluate the trained model
    evaluate_model(trained_model, val_loader)

else:
    print(
        "\nSkipping model training and evaluation due to data loading or splitting issues."
    )


# %% [markdown]
# ## Inference Function for Probabilistic Output
# Defines a function to perform inference on partial weather data up to a given date.
# It runs the trained model multiple times with dropout enabled to generate a sample
# of possible yield outcomes, which can then be visualized as a histogram.


# %%
def predict_distribution(
    model,
    data_lake_dir,
    year,
    county_fips,
    crop_name,
    inference_date,  # Example: 'YYYY-MM-DD' (e.g., '2023-05-15')
    fips_to_id_mapping,
    num_samples=200,  # Number of runs for dropout sampling (more samples give a better histogram)
    device=None,
):
    """
    Predicts a yield distribution for a specific year, county, crop, and inference date.

    Loads weather data up to the inference_date, formats input, and runs the trained
    model multiple times with dropout enabled (Monte Carlo Dropout) to approximate
    the predictive distribution.

    Args:
        model (torch.nn.Module): The trained model (must be in evaluation mode initially).
        data_lake_dir (str): Path to the data lake root directory.
        year (int): The year for prediction.
        county_fips (str): The FIPS code (string) for the county.
        crop_name (str): The name of the crop (used for messaging).
        inference_date (str): The date up to which weather data is available (YYYY-MM-DD).
        fips_to_id_mapping (dict): Dictionary mapping FIPS codes (str) to integer IDs,
                                   obtained from the training dataset.
        num_samples (int): Number of forward passes with dropout to sample the distribution.
        device (torch.device, optional): The device (CPU/GPU) to use for inference.
                                         Defaults to automatically detecting GPU or using CPU.

    Returns:
        tuple: (predicted_yields_samples, bins, hist_counts) or (None, None, None) if error occurs.
               predicted_yields_samples (list): List of individual predicted yields from each sample run.
               bins (numpy.ndarray): Edges of the histogram bins.
               hist_counts (numpy.ndarray): Counts of samples falling into each bin.
    """
    year_str = str(year)
    crop_name_lower = crop_name.lower()
    # Determine device
    device = (
        device
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)  # Ensure model is on the correct device

    print(f"\n--- Performing Inference ---")
    print(f"Inference Date: {inference_date}")
    print(f"County FIPS: {county_fips}")
    print(f"Crop: {crop_name}")
    print(f"Year: {year}")
    print(f"Number of samples for distribution: {num_samples}")
    print(f"Device: {device}")

    # --- 1. Load and Process Partial Weather Data ---
    fips_folder_path = Path(data_lake_dir) / county_fips
    year_folder_path = fips_folder_path / year_str
    weather_csv_path = year_folder_path / f"WeatherTimeSeries{year_str}.csv"

    if not weather_csv_path.exists():
        print(
            f"Error: Weather CSV not found at {weather_csv_path} for FIPS {county_fips}, year {year_str}"
        )
        return None, None, None

    try:
        df_weather = pd.read_csv(weather_csv_path)
    except Exception as e:
        print(f"Error reading weather CSV {weather_csv_path}: {e}")
        return None, None, None

    # Filter weather data up to the specified inference date
    try:
        # Ensure date columns are present
        if not all(col in df_weather.columns for col in ["Year", "Month", "Day"]):
            print(
                f"Error: Weather CSV for FIPS {county_fips}, Year {year_str} is missing Year, Month, or Day columns."
            )
            return None, None, None

        # Create a datetime column for filtering
        df_weather["Date"] = pd.to_datetime(df_weather[["Year", "Month", "Day"]])
        inference_date_dt = pd.to_datetime(
            inference_date
        )  # Convert inference date string to datetime object

        # Filter rows where the date is less than or equal to the inference date
        df_partial_weather = df_weather[
            df_weather["Date"] <= inference_date_dt
        ].copy()  # Use .copy()

        # Optional: Filter to include only relevant months if inference date is later than April 1
        # This aligns inference data structure somewhat with training (April-Oct focus)
        # However, filtering *only* to <= inference_date allows mid-season predictions correctly
        # Keeping all available data up to the inference date is the goal here.

        # Drop date-related columns and others not used as features
        cols_to_drop = ["Year", "Month", "Day", "Date"]
        existing_cols_to_drop = [
            col for col in cols_to_drop if col in df_partial_weather.columns
        ]
        df_partial_weather = df_partial_weather.drop(
            columns=existing_cols_to_drop, errors="ignore"
        )

        # Ensure all weather columns have numeric types and drop NaNs introduced by coerce
        for col in df_partial_weather.columns:
            df_partial_weather[col] = pd.to_numeric(
                df_partial_weather[col], errors="coerce"
            )
        df_partial_weather.dropna(
            axis=1, how="all", inplace=True
        )  # Drop columns all NaN
        initial_rows = len(df_partial_weather)
        df_partial_weather.dropna(
            axis=0, how="any", inplace=True
        )  # Drop rows with any NaN
        if len(df_partial_weather) < initial_rows:
            print(
                f"Warning: Dropped {initial_rows - len(df_partial_weather)} rows with NaN weather data up to {inference_date}."
            )

    except Exception as e:
        print(f"Error processing weather data for date filtering: {e}")
        return None, None, None

    if df_partial_weather.empty:
        print(
            f"Error: No valid weather data found up to {inference_date} for FIPS {county_fips}, year {year_str}"
        )
        return None, None, None

    # Get the number of weather features expected by the model (from training dataset)
    # This assumes the order and number of features in the inference data matches training
    # It's crucial that the columns match the order from CropYieldDataset processing
    # A safer approach would store the feature names/order during training setup
    if trained_model is None:
        print("Error: Model is not trained.")
        return None, None, None
    # Assuming input_dim from training is still available or can be retrieved
    expected_input_dim = model.lstm.input_size - model.fips_embedding.embedding_dim
    if df_partial_weather.shape[1] != expected_input_dim:
        print(
            f"Error: Weather data for inference has {df_partial_weather.shape[1]} features, but model expects {expected_input_dim}. Check column consistency."
        )
        # You might need to reorder/select columns here if they aren't guaranteed to match
        return None, None, None

    # Convert the partial weather data to a PyTorch tensor
    # unsqueeze(0) adds the batch dimension (batch size = 1 for single inference sample)
    weather_tensor = (
        torch.tensor(df_partial_weather.values, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )  # Shape: (1, T, features)

    # --- 2. Get County ID ---
    # Look up the integer ID for the target county FIPS code using the mapping from training
    if county_fips not in fips_to_id_mapping:
        print(
            f"Error: County FIPS {county_fips} not found in the training mapping ({len(fips_to_id_mapping)} counties trained)."
        )
        print("The model was not trained on this specific county.")
        # Handle this case: could return None, predict a regional average, etc.
        # For now, we'll return None
        return None, None, None

    fips_id = fips_to_id_mapping[county_fips]
    # Convert the single FIPS ID to a tensor with batch dimension
    fips_tensor = torch.tensor([fips_id], dtype=torch.long).to(device)  # Shape: (1,)

    # --- 3. Run Model Multiple Times with Dropout (Monte Carlo Dropout) ---
    model.train()  # Set model to training mode to enable dropout layers
    # This is the key step for Monte Carlo Dropout sampling

    predicted_yields = []  # List to collect predictions from each sample run
    with torch.no_grad():  # Ensure no gradients are computed during sampling
        # Run the model multiple times for the same input
        for i in tqdm.tqdm(range(num_samples), desc="Sampling Predictions"):
            # Pass weather and fips tensor (batch size 1)
            prediction = model(weather_tensor, fips_tensor)
            # .item() gets the scalar value from the 1-element tensor
            predicted_yields.append(prediction.item())

    model.eval()  # Set model back to standard evaluation mode (dropout off)

    # --- 4. Generate Histogram Data ---
    # Calculate histogram bins and counts from the collected predictions
    if not predicted_yields:  # Check if any predictions were collected
        print("Error: No predictions were generated.")
        return None, None, None

    # Use the range of predicted yields to define histogram bins
    min_yield = min(predicted_yields)
    max_yield = max(predicted_yields)

    # Handle edge case where all predictions are the same
    if min_yield == max_yield:
        print(
            f"Warning: All {num_samples} predictions are the same value: {min_yield:.2f}"
        )
        # Create a single bin for the histogram
        bins = np.array(
            [min_yield - 1, max_yield + 1]
        )  # A small range around the value
        hist_counts = np.array([num_samples])
    else:
        # Create bins - e.g., 20 bins between min and max predictions
        num_bins = 20  # Can be a parameter
        bins = np.linspace(min_yield, max_yield, num_bins + 1)
        # Calculate histogram counts. density=False gives counts per bin.
        hist_counts, bin_edges = np.histogram(
            predicted_yields, bins=bins, density=False
        )
        bins = bin_edges  # Use the actual bin edges calculated by numpy

    print(f"Generated histogram data from {num_samples} samples.")
    print(f"Predicted Yield Range: [{min_yield:.2f}, {max_yield:.2f}]")
    print(f"Mean Predicted Yield: {np.mean(predicted_yields):.2f}")
    print(f"Std Dev of Predicted Yield: {np.std(predicted_yields):.2f}")

    # Return the raw samples, bin edges, and counts
    return predicted_yields, bins, hist_counts


# %% [markdown]
# ## Inference Execution
# Example of how to use the `predict_distribution` function with a trained model
# and plot the resulting histogram.

# %%
# --- How to use the inference function ---
# Ensure trained_model and fips_to_id_mapping_from_training are available from the training block
# %% [markdown]
# ## Inference Execution
# Example of how to use the `predict_distribution` function with a trained model
# and plot the resulting histogram.

# %%
# --- How to use the inference function ---
# Ensure trained_model and fips_to_id_mapping are available from the training block

# The variable fips_to_id_mapping is defined in the "Prepare DataLoaders" block.
# The variable trained_model is defined in the "Model Initialization and Training" block.
# We check if the model was successfully trained before attempting inference.
if trained_model is not None:
    # Set your data root path
    data_lake_root = "data/data_lake_organized"
    # Specify the year, county, crop, and the date up to which weather data is available
    # Ensure target_county_fips is one of the FIPS codes printed during dataset loading.
    target_year = 2022
    target_county_fips = "19109" # Example FIPS (replace with a FIPS code present in your data)
    target_crop = "corn"
    current_date = "2022-06-15" # Date up to which weather is available (YYYY-MM-DD)

    # Perform the probabilistic inference
    # Pass the correct FIPS mapping variable: fips_to_id_mapping
    predicted_yields_samples, bins, hist_counts = predict_distribution(
        trained_model,
        data_lake_root,
        target_year,
        target_county_fips,
        target_crop,
        current_date,
        fips_to_id_mapping, # <<< Use the correct variable name here
        num_samples=500, # Increased samples for a smoother histogram
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # Specify device
    )

    # --- Visualize or Output Results ---
    if predicted_yields_samples is not None:
        # Plot the histogram if data was successfully generated
        plt.figure(figsize=(12, 7))
        # Use plt.hist for automatic binning and density calculation if desired,
        # or plt.bar with calculated bins/counts
        plt.bar(bins[:-1], hist_counts, width=np.diff(bins), edgecolor='black', alpha=0.7, align='edge')

        plt.xlabel(f"Predicted Yield ({target_crop.capitalize()} bushels)")
        plt.ylabel("Frequency (Count of Samples)")
        plt.title(f"Predicted Yield Distribution for {target_crop.capitalize()} in FIPS {target_county_fips}, Year {target_year} (Data up to {current_date})")
        plt.grid(axis='y', alpha=0.5)
        plt.tight_layout()


        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"  Mean: {np.mean(predicted_yields_samples):.2f}")
        print(f"  Median: {np.median(predicted_yields_samples):.2f}")
        print(f"  Standard Deviation: {np.std(predicted_yields_samples):.2f}")
        print(f"  Min: {min(predicted_yields_samples):.2f}")
        print(f"  Max: {max(predicted_yields_samples):.2f}")

        # Optional: Add a vertical line for the mean prediction
        plt.axvline(np.mean(predicted_yields_samples), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {np.mean(predicted_yields_samples):.2f}')
        plt.legend()

        plt.show() # Display the plot

        # Print histogram data points (optional)
        print("\nHistogram Data (Bin Range, Count):")
        # Ensure bin edges and counts match in length for iteration
        if len(bins) > 1 and len(hist_counts) == len(bins) -1:
            for i in range(len(hist_counts)):
                print(f"  [{bins[i]:.2f} - {bins[i+1]:.2f}), Count: {hist_counts[i]}")
        elif len(bins) == 2 and len(hist_counts) == 1: # Case for single bin (all predictions same)
             print(f"  [{bins[0]:.2f} - {bins[1]:.2f}), Count: {hist_counts[0]}")
        else:
             print("  Could not format histogram data points.")


    else:
        print("\nInference failed. Cannot generate histogram.")

else:
    print("\nSkipping inference. Model was not successfully trained or data was insufficient.")


# %%