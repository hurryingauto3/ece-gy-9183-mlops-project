# %% [markdown]
# ## Imports
# Import necessary libraries: torch, pandas, json, os, pathlib, numpy, matplotlib, random.
# torch.nn is for neural network layers, torch.utils.data for data handling.

# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn.utils.rnn # Needed for pad_sequence

import pandas as pd
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tqdm # For progress bars
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random # Needed for random selection of years
import mlflow.pytorch


# %% [markdown]
# ## Dataset Class (Modified)
# Defines a custom Dataset to load weather time series and annual yield data.
# Modified to store the year along with each sample and provide methods to access samples by year.

# %%
# class CropYieldDataset(Dataset):
#     """
#     Loads crop yield data along with corresponding weather time series for specific counties and years.
#     Modified to store the year for splitting.

#     Organizes data from a specified directory structure:
#     data_lake_dir/
#     ├── FIPS_CODE_1/
#     │   ├── crop_name.json (contains yearly yield data for this FIPS)
#     │   ├── YEAR_1/
#     │   │   └── WeatherTimeSeriesYEAR_1.csv (daily weather data)
#     │   ├── YEAR_2/
#     │   │   └── WeatherTimeSeriesYEAR_2.csv
#     │   └── ...
#     └── FIPS_CODE_2/
#         └── ...
#     """
#     def __init__(self, data_lake_dir="data/data_lake_organized", crop_name="corn", transform=None):
#         # Stores samples as tuples: (weather_tensor, yield_target, fips_id, year_str)
#         self._samples_with_year = []
#         self.transform = transform # Optional transformations on weather data
#         self.crop_name = crop_name.lower()

#         # Dictionaries to map FIPS string codes to integer IDs and vice-versa
#         self.fips_to_id = {}
#         self.id_to_fips = {}
#         self._next_fips_id = 0 # Counter for assigning unique integer IDs

#         data_lake_path = Path(data_lake_dir)
#         if not data_lake_path.exists():
#             raise FileNotFoundError(f"Data lake directory not found at: {data_lake_dir}")

#         # Dictionary to store indices of samples grouped by year
#         self._samples_by_year_indices = {}
#         self._current_sample_index = 0

#         # Iterate through each FIPS code folder in the data lake directory
#         fips_folders = [f for f in data_lake_path.iterdir() if f.is_dir()]

#         if not fips_folders:
#              print(f"Warning: No FIPS folders found in {data_lake_dir}.")


#         for fips_folder in tqdm.tqdm(fips_folders, desc=f"Loading Data for {self.crop_name}"):
#             fips_code = fips_folder.name # Folder name is the FIPS code (string)

#             # Assign a unique integer ID if this FIPS code hasn't been seen before
#             if fips_code not in self.fips_to_id:
#                 self.fips_to_id[fips_code] = self._next_fips_id
#                 self.id_to_fips[self._next_fips_id] = fips_code
#                 self._next_fips_id += 1

#             fips_id = self.fips_to_id[fips_code]

#             # Check if the crop's yield JSON file exists for this FIPS
#             crop_json_path = fips_folder / f"{self.crop_name}.json"
#             if not crop_json_path.exists():
#                 continue # Skip this FIPS if no yield data for the target crop

#             # Load the yield data for this crop and FIPS
#             try:
#                 with open(crop_json_path, 'r') as f:
#                     yield_data = json.load(f)
#             except json.JSONDecodeError:
#                  print(f"Warning: Could not decode JSON from {crop_json_path} for FIPS {fips_code}. Skipping.")
#                  continue # Skip if JSON is invalid
#             except Exception as e:
#                  print(f"Warning: Error reading {crop_json_path} for FIPS {fips_code}: {e}. Skipping.")
#                  continue # Skip on other file errors

#             # Iterate through year folders within the FIPS folder
#             year_folders = [y for y in year_folder.iterdir() if y.is_dir()]
#             year_folders.sort() # Sort years to process chronologically (optional but good practice)

#             for year_folder in year_folders:
#                 year_str = year_folder.name # Year is a string from folder name

#                 # Check if yield data exists for this specific year
#                 if year_str not in yield_data or 'yield' not in yield_data[year_str]:
#                      continue # Skip this year if no yield data

#                 # Check if the weather CSV exists for this year and FIPS
#                 weather_csv = year_folder / f"WeatherTimeSeries{year_str}.csv"
#                 if not weather_csv.exists():
#                     continue # Skip this year if weather data is missing

#                 # Load the weather data
#                 try:
#                     df = pd.read_csv(weather_csv)
#                 except pd.errors.EmptyDataError:
#                      print(f"Warning: Weather CSV empty for FIPS {fips_code}, Year {year_str}. Skipping.")
#                      continue
#                 except Exception as e:
#                      print(f"Warning: Error reading {weather_csv} for FIPS {fips_code}, Year {year_str}: {e}. Skipping.")
#                      continue


#                 # Filter weather data to the growing season (April to October)
#                 df_season = df[(df['Month'] >= 4) & (df['Month'] <= 10)].copy() # Use .copy() to avoid SettingWithCopyWarning

#                 if df_season.empty:
#                     continue # Skip if no weather data in the target range

#                 # Drop non-weather columns like Year, Month, Day
#                 cols_to_drop = ['Year', 'Month', 'Day']
#                 existing_cols_to_drop = [col for col in cols_to_drop if col in df_season.columns]
#                 df_season = df_season.drop(columns=existing_cols_to_drop, errors='ignore')

#                 # Ensure all weather columns have numeric types
#                 for col in df_season.columns:
#                      df_season[col] = pd.to_numeric(df_season[col], errors='coerce')

#                 # Drop any columns that became all NaN after conversion, or have very few non-NaN values
#                 df_season.dropna(axis=1, how='all', inplace=True)
#                 # Optional: Drop columns with too many NaNs (e.g., more than 10% missing per year)
#                 # thresh=int(len(df_season)*0.9) means keep column if it has at least 90% non-NaN values
#                 # df_season.dropna(axis=1, thresh=int(len(df_season)*0.9), inplace=True)


#                 # Drop rows with any NaN values (missing days within the season)
#                 # Or impute NaNs if appropriate (e.g., forward fill, mean)
#                 initial_rows = len(df_season)
#                 df_season.dropna(axis=0, how='any', inplace=True)
#                 if len(df_season) < initial_rows:
#                      # print(f"Warning: Dropped {initial_rows - len(df_season)} rows with NaN weather data for FIPS {fips_code}, Year {year_str}. {len(df_season)} rows remaining.")
#                      if df_season.empty:
#                           # print(f"Skipping FIPS {fips_code}, Year {year_str}: No valid weather data rows left.")
#                           continue


#                 # Convert the cleaned weather data to a PyTorch tensor
#                 weather_tensor = torch.tensor(df_season.values, dtype=torch.float32)

#                 # Get the yield target and convert to tensor
#                 yield_target = torch.tensor(yield_data[year_str]['yield'], dtype=torch.float32)

#                 # Store the sample along with its year
#                 self._samples_with_year.append((weather_tensor, yield_target, fips_id, year_str))

#                 # Store the index of this sample, grouped by year
#                 if year_str not in self._samples_by_year_indices:
#                     self._samples_by_year_indices[year_str] = []
#                 self._samples_by_year_indices[year_str].append(self._current_sample_index)
#                 self._current_sample_index += 1


#         print(f"\nFinished loading data.")
#         print(f"Loaded {len(self._samples_with_year)} total samples for crop '{self.crop_name}'.")
#         print(f"Found {len(self.fips_to_id)} unique FIPS codes.")
#         print(f"Found data for years: {sorted(self._samples_by_year_indices.keys())}")


#     def __len__(self):
#         """Returns the total number of samples loaded across all years and counties."""
#         return len(self._samples_with_year)

#     def __getitem__(self, idx):
#         """Retrieves a single sample by index. Returns weather, yield, fips_id (excluding year)."""
#         # Retrieve the sample tuple
#         weather_tensor, yield_target, fips_id, year_str = self._samples_with_year[idx]
#         # Apply transform only to the weather tensor if specified
#         if self.transform:
#             weather_tensor = self.transform(weather_tensor)
#         # Return the sample components needed for the model (excluding the year string)
#         return weather_tensor, yield_target, fips_id

#     def get_fips_mapping(self):
#         """Returns the dictionaries mapping FIPS codes to IDs and vice-versa."""
#         return self.fips_to_id, self.id_to_fips

#     def get_num_fips(self):
#         """Returns the number of unique FIPS codes found."""
#         return len(self.fips_to_id)

#     def get_years(self):
#         """Returns a sorted list of all unique years present in the dataset."""
#         return sorted(self._samples_by_year_indices.keys())

#     def get_sample_indices_by_year(self):
#         """Returns a dictionary mapping year (str) to a list of sample indices for that year."""
#         return self._samples_by_year_indices


class CropYieldDataset(Dataset):
    """
    Loads crop yield data along with corresponding weather time series for specific counties and years.
    Modified to store the year for splitting.

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
    def __init__(self, data_lake_dir="../data/data_lake_organized", crop_name="corn", transform=None):
        # Stores samples as tuples: (weather_tensor, yield_target, fips_id, year_str)
        self._samples_with_year = []
        self.transform = transform # Optional transformations on weather data
        self.crop_name = crop_name.lower()

        # Dictionaries to map FIPS string codes to integer IDs and vice-versa
        self.fips_to_id = {}
        self.id_to_fips = {}
        self._next_fips_id = 0 # Counter for assigning unique integer IDs

        data_lake_path = Path(data_lake_dir)
        if not data_lake_path.exists():
            raise FileNotFoundError(f"Data lake directory not found at: {data_lake_dir}")

        # Dictionary to store indices of samples grouped by year
        self._samples_by_year_indices = {}
        self._current_sample_index = 0

        # Iterate through each FIPS code folder in the data lake directory
        fips_folders = [f for f in data_lake_path.iterdir() if f.is_dir()]

        if not fips_folders:
             print(f"Warning: No FIPS folders found in {data_lake_dir}.")
             # Consider raising an error here if no data means the dataset is unusable

        # Use sorted fips folders for consistent processing order
        for fips_folder in tqdm.tqdm(sorted(fips_folders), desc=f"Loading Data for {self.crop_name}"):
            fips_code = fips_folder.name # Folder name is the FIPS code (string)

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
                continue # Skip this FIPS if no yield data for the target crop

            # Load the yield data for this crop and FIPS
            try:
                with open(crop_json_path, 'r') as f:
                    yield_data = json.load(f)
            except json.JSONDecodeError:
                 print(f"Warning: Could not decode JSON from {crop_json_path} for FIPS {fips_code}. Skipping.")
                 continue # Skip if JSON is invalid
            except Exception as e:
                 print(f"Warning: Error reading {crop_json_path} for FIPS {fips_code}: {e}. Skipping.")
                 continue # Skip on other file errors

            # Iterate through year folders within the FIPS folder
            # THIS IS THE CORRECTED LINE: Iterate over the *contents* of fips_folder
            year_folders = [y for y in fips_folder.iterdir() if y.is_dir()]
            year_folders.sort() # Sort years to process chronologically (optional but good practice)

            if not year_folders:
                # print(f"Warning: No year folders found for FIPS {fips_code}.") # Optional detailed log
                continue # Skip if no year folders found

            for year_folder in year_folders:
                year_str = year_folder.name # Year is a string from folder name

                # Check if yield data exists for this specific year
                if year_str not in yield_data or 'yield' not in yield_data[year_str]:
                     continue # Skip this year if no yield data

                # Check if the weather CSV exists for this year and FIPS
                weather_csv = year_folder / f"WeatherTimeSeries{year_str}.csv"
                if not weather_csv.exists():
                    continue # Skip this year if weather data is missing

                # Load the weather data
                try:
                    df = pd.read_csv(weather_csv)
                except pd.errors.EmptyDataError:
                     print(f"Warning: Weather CSV empty for FIPS {fips_code}, Year {year_str}. Skipping.")
                     continue
                except Exception as e:
                     print(f"Warning: Error reading {weather_csv} for FIPS {fips_code}, Year {year_str}: {e}. Skipping.")
                     continue


                # Filter weather data to the growing season (April to October)
                df_season = df[(df['Month'] >= 4) & (df['Month'] <= 10)].copy() # Use .copy() to avoid SettingWithCopyWarning

                if df_season.empty:
                    continue # Skip if no weather data in the target range

                # Drop non-weather columns like Year, Month, Day
                cols_to_drop = ['Year', 'Month', 'Day']
                existing_cols_to_drop = [col for col in cols_to_drop if col in df_season.columns]
                df_season = df_season.drop(columns=existing_cols_to_drop, errors='ignore')

                # Ensure all weather columns have numeric types
                for col in df_season.columns:
                     df_season[col] = pd.to_numeric(df_season[col], errors='coerce')

                # Drop any columns that became all NaN after conversion, or have very few non-NaN values
                df_season.dropna(axis=1, how='all', inplace=True)
                # Optional: Drop columns with too many NaNs (e.g., more than 10% missing per year)
                # thresh=int(len(df_season)*0.9) means keep column if it has at least 90% non-NaN values
                # df_season.dropna(axis=1, thresh=int(len(df_season)*0.9), inplace=True)


                # Drop rows with any NaN values (missing days within the season)
                # Or impute NaNs if appropriate (e.g., forward fill, mean)
                initial_rows = len(df_season)
                df_season.dropna(axis=0, how='any', inplace=True)
                if len(df_season) < initial_rows:
                     # print(f"Warning: Dropped {initial_rows - len(df_season)} rows with NaN weather data for FIPS {fips_code}, Year {year_str}. {len(df_season)} rows remaining.")
                     if df_season.empty:
                          # print(f"Skipping FIPS {fips_code}, Year {year_str}: No valid weather data rows left.")
                          continue


                # Check if any weather features remain after dropping columns/rows
                if df_season.shape[1] == 0:
                    print(f"Warning: No weather features remaining for FIPS {fips_code}, Year {year_str} after cleaning. Skipping.")
                    continue


                # Convert the cleaned weather data to a PyTorch tensor
                weather_tensor = torch.tensor(df_season.values, dtype=torch.float32)

                # Get the yield target and convert to tensor
                yield_target = torch.tensor(yield_data[year_str]['yield'], dtype=torch.float32)

                # Store the sample along with its year
                self._samples_with_year.append((weather_tensor, yield_target, fips_id, year_str))

                # Store the index of this sample, grouped by year
                if year_str not in self._samples_by_year_indices:
                    self._samples_by_year_indices[year_str] = []
                self._samples_by_year_indices[year_str].append(self._current_sample_index)
                self._current_sample_index += 1


        print(f"\nFinished loading data.")
        print(f"Loaded {len(self._samples_with_year)} total samples for crop '{self.crop_name}'.")
        print(f"Found {len(self.fips_to_id)} unique FIPS codes.")
        print(f"Found data for years: {sorted(self._samples_by_year_indices.keys())}")

    # ... rest of the CropYieldDataset methods remain the same ...
    def __len__(self):
        """Returns the total number of samples loaded across all years and counties."""
        return len(self._samples_with_year)

    def __getitem__(self, idx):
        """Retrieves a single sample by index. Returns weather, yield, fips_id (excluding year)."""
        # Retrieve the sample tuple
        weather_tensor, yield_target, fips_id, year_str = self._samples_with_year[idx]
        # Apply transform only to the weather tensor if specified
        if self.transform:
            weather_tensor = self.transform(weather_tensor)
        # Return the sample components needed for the model (excluding the year string)
        return weather_tensor, yield_target, fips_id

    def get_fips_mapping(self):
        """Returns the dictionaries mapping FIPS codes to IDs and vice-versa."""
        return self.fips_to_id, self.id_to_fips

    def get_num_fips(self):
        """Returns the number of unique FIPS codes found."""
        return len(self.fips_to_id)

    def get_years(self):
        """Returns a sorted list of all unique years present in the dataset."""
        return sorted(self._samples_by_year_indices.keys())

    def get_sample_indices_by_year(self):
        """Returns a dictionary mapping year (str) to a list of sample indices for that year."""
        return self._samples_by_year_indices

# %%
def save_model_to_mlflow(model, input_example, model_name="AgriYieldPredictor"):
    """
    Logs the trained model to MLflow for serving.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        input_example (tuple): A sample input for the model (e.g., weather_tensor, fips_tensor).
        model_name (str): Name of the model in the MLflow Model Registry.
    """
    # Log the model to MLflow
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="pytorch_model",  # Path within the MLflow run
        registered_model_name=model_name,  # Register the model in the MLflow Model Registry
        input_example=input_example  # Provide an example input for inference
    )
    print(f"Model logged to MLflow under the name '{model_name}'.")


# %% [markdown]
# ## Collate Function
# Custom `collate_fn` for the DataLoader. This is necessary because weather sequences
# have varying lengths and need to be padded to create batches. It also correctly
# stacks the yield targets and FIPS IDs. This remains the same as before.

# %%
def collate_fn(batch):
    """
    Collates a batch of samples. Pads weather sequences and stacks targets and FIPS IDs.

    Args:
        batch (list): A list of samples, where each sample is a tuple
                      (weather_tensor, yield_target, fips_id) - note: year is *not*
                      returned by __getitem__ for this collate_fn.

    Returns:
        tuple: Padded weather tensor batch, stacked yield target batch, stacked FIPS ID batch.
    """
    # Separate the elements of the batch
    # batch is a list of tuples: [(weather_tensor_1, yield_target_1, fips_id_1), ...]
    xs = [item[0] for item in batch]      # List of weather_tensors (varying lengths)
    ys = [item[1] for item in batch]      # List of yield_targets (scalar tensors)
    fips_ids = [item[2] for item in batch] # List of fips_ids (Python integers)

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
    fips_ids_stacked = torch.stack(fips_ids_tensors) # Stack into a batch tensor (batch_size,)

    # Return the batch as a tuple of tensors
    return xs_padded, ys_stacked, fips_ids_stacked


# %% [markdown]
# ## Model Definition
# Defines the `LSTMTCNRegressor` neural network model.
# This model uses an Embedding layer for county FIPS IDs, an LSTM for processing weather sequences,
# a TCN for capturing local temporal features, and a final Linear layer to predict yield.
# This remains the same as before.

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
    def __init__(self, input_dim, num_fips, fips_embedding_dim=16, hidden_dim=64, lstm_layers=1, tcn_channels=[64, 32], dropout_rate=0.1):
        super(LSTMTCNRegressor, self).__init__()

        # Embedding layer for FIPS codes
        self.fips_embedding = nn.Embedding(num_fips, fips_embedding_dim)

        # Input to LSTM will be weather_features + fips_embedding_dim
        lstm_input_dim = input_dim + fips_embedding_dim

        # LSTM layer to process the sequence of combined weather and FIPS features
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=lstm_layers, batch_first=True, dropout=dropout_rate if lstm_layers > 1 else 0)

        # TCN part using 1D Convolutions
        tcn_layers = []
        in_channels = hidden_dim
        for i, out_channels in enumerate(tcn_channels):
             tcn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
             tcn_layers.append(nn.ReLU())
             if dropout_rate > 0: # Apply dropout after ReLU
                 tcn_layers.append(nn.Dropout(dropout_rate))
             in_channels = out_channels

        self.tcn = nn.Sequential(*tcn_layers)

        # Adaptive pooling after TCN to get a fixed-size vector
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Final fully connected layer
        self.fc = nn.Linear(tcn_channels[-1], 1)

        self.dropout_rate = dropout_rate # Store dropout rate

    def forward(self, x_weather, fips_ids):
        """
        Forward pass of the model.

        Args:
            x_weather (torch.Tensor): Padded weather time series data (batch, time_steps, weather_features).
            fips_ids (torch.Tensor): Batch of FIPS integer IDs (batch,).

        Returns:
            torch.Tensor: Predicted yield for each sample in the batch (batch,).
        """
        # Get county embeddings and expand across time dimension
        fips_emb = self.fips_embedding(fips_ids)
        fips_emb_expanded = fips_emb.unsqueeze(1).repeat(1, x_weather.size(1), 1)

        # Concatenate weather features and county embeddings
        x_combined = torch.cat([x_weather, fips_emb_expanded], dim=-1)

        # Pass through LSTM
        out, _ = self.lstm(x_combined)

        # Permute for Conv1D
        out = out.permute(0, 2, 1)

        # Pass through TCN
        out = self.tcn(out)

        # Apply pooling
        out = self.pooling(out)

        # Squeeze the last dimension
        out = out.squeeze(-1)

        # Pass through final FC layer
        out = self.fc(out)

        # Squeeze the last dimension for scalar output
        return out.squeeze(-1)


# %% [markdown]
# ## Data Loading and Splitting (Year-based)
# Instantiate the dataset and split samples into training, validation, and holdout sets based on year.
# Create DataLoaders for the training and validation sets.

# %%
def get_dataloaders_by_year(dataset, holdout_year, val_year_ratio=0.2, batch_size=32):
    """
    Splits the dataset samples into training, validation, and holdout sets based on year.
    Returns DataLoaders for training and validation.

    Args:
        dataset (CropYieldDataset): The dataset instance.
        holdout_year (int or str): The year(s) to hold out for testing/inference. Can be a single year (str)
                                   or a list of years (list of str).
        val_year_ratio (float): The proportion of *remaining* years (after holding out) to use for validation.
                                If < 1, it's a ratio. If >= 1, it's treated as the number of validation years.
        batch_size (int): The batch size for the DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, holdout_dataset)
               train_loader (DataLoader): DataLoader for training data.
               val_loader (DataLoader): DataLoader for validation data.
               holdout_dataset (Subset): Subset dataset for holdout year(s).

    Raises:
        ValueError: If the dataset is empty, holdout year not found, or not enough years for splitting.
        TypeError: If holdout_year is not a string or list of strings.
    """
    if len(dataset) == 0:
         raise ValueError("Dataset is empty. Cannot create dataloaders.")

    # Ensure holdout_year is in the correct format (string or list of strings)
    if isinstance(holdout_year, int):
         holdout_year = str(holdout_year) # Convert int year to string
    if isinstance(holdout_year, str):
         holdout_years = [holdout_year]
    elif isinstance(holdout_year, list):
         holdout_years = [str(y) for y in holdout_year] # Ensure all are strings
    else:
         raise TypeError("holdout_year must be an integer, string, or a list of integers/strings.")


    all_years = dataset.get_years() # Get all unique years as sorted strings
    samples_by_year_indices = dataset.get_sample_indices_by_year() # Get dict: year -> list of indices

    # Validate holdout year(s)
    for year in holdout_years:
        if year not in all_years:
            raise ValueError(f"Holdout year '{year}' not found in the dataset. Available years: {all_years}")

    # Separate holdout years and remaining years
    remaining_years = [year for year in all_years if year not in holdout_years]

    if not remaining_years:
        raise ValueError(f"No years left for training/validation after holding out {holdout_years}.")

    # Split remaining years into training and validation years
    random.shuffle(remaining_years) # Shuffle years before splitting

    if val_year_ratio >= 1: # Treat ratio as a fixed number of years
        num_val_years = int(val_year_ratio) # Use this number of years for validation
        if num_val_years >= len(remaining_years):
             raise ValueError(f"Number of validation years ({num_val_years}) is >= remaining years ({len(remaining_years)}). Cannot create training set.")
        val_years = remaining_years[:num_val_years]
        train_years = remaining_years[num_val_years:]
    else: # Treat ratio as a proportion
        num_val_years = max(1, int(len(remaining_years) * val_year_ratio)) # Ensure at least 1 val year if possible
        if len(remaining_years) - num_val_years == 0:
             # If ratio makes train set empty, maybe take one less val year if possible
             if num_val_years > 1:
                 num_val_years -= 1
             else:
                 raise ValueError(f"Validation ratio {val_year_ratio} leaves no years for training after holding out {holdout_years}.")

        val_years = remaining_years[:num_val_years]
        train_years = remaining_years[num_val_years:]


    print(f"\nSplitting data by year:")
    print(f"Holdout year(s): {holdout_years}")
    print(f"Training year(s): {sorted(train_years)}")
    print(f"Validation year(s): {sorted(val_years)}")

    # Collect sample indices for each split
    train_indices = [idx for year in train_years for idx in samples_by_year_indices[year]]
    val_indices = [idx for year in val_years for idx in samples_by_year_indices[year]]
    holdout_indices = [idx for year in holdout_years for idx in samples_by_year_indices[year]]

    if not train_indices:
         raise ValueError("Training set is empty after splitting years.")
    if not val_indices:
         raise ValueError("Validation set is empty after splitting years.")
    if not holdout_indices:
         print(f"Warning: Holdout set for year(s) {holdout_years} is empty.") # Might happen if the year has no data for the crop/fips


    # Create Subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    holdout_dataset = Subset(dataset, holdout_indices)

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Holdout samples: {len(holdout_dataset)}")


    # Create DataLoaders for train and validation using the custom collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Training DataLoader created with {len(train_loader)} batches.")
    print(f"Validation DataLoader created with {len(val_loader)} batches.")


    return train_loader, val_loader, holdout_dataset

# %% [markdown]
# ## Training Function
# Defines the `train_model` function to train the neural network.
# It includes moving data and model to device (GPU/CPU), defining loss and optimizer,
# and iterating through epochs with training and validation steps. This remains the same.

# %%
def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-3):
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
    model = model.to(device) # Move model to the selected device

    # Mean Squared Error Loss for regression
    criterion = nn.MSELoss()
    # Adam optimizer for updating model weights
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf') # Track best validation loss for potential early stopping/saving

    # Loop through epochs
    for epoch in tqdm.tqdm(range(num_epochs), desc="Training Progress"):
        model.train() # Set model to training mode (enables dropout etc.)
        epoch_loss = 0 # Accumulate loss for the current epoch

        # Iterate over training batches
        for x_batch, y_batch, fips_batch in train_loader:
            # Move batch data to the selected device
            x_batch, y_batch, fips_batch = x_batch.to(device), y_batch.to(device), fips_batch.to(device)

            # Zero the gradients from the previous iteration
            optimizer.zero_grad()

            # Perform forward pass: get model predictions
            y_pred = model(x_batch, fips_batch) # Pass weather sequences and FIPS IDs

            # Calculate the loss
            loss = criterion(y_pred, y_batch)

            # Perform backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Perform a single optimization step: update model parameters
            optimizer.step()

            # Accumulate the loss
            epoch_loss += loss.item()

        # --- Validation Step ---
        model.eval() # Set model to evaluation mode (disables dropout etc.)
        val_loss = 0
        # Disable gradient calculation for validation
        with torch.no_grad():
            for x_val, y_val, fips_val in val_loader:
                x_val, y_val, fips_val = x_val.to(device), y_val.to(device), fips_val.to(device)
                y_pred = model(x_val, fips_val) # Get predictions
                val_loss += criterion(y_pred, y_val).item() # Accumulate validation loss

        # Calculate average losses for the epoch
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Print epoch statistics using tqdm.write to avoid interfering with the progress bar
        tqdm.tqdm.write(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss} - Val Loss: {avg_val_loss}")
        # Update the progress bar description with the current validation loss
        tqdm.tqdm.write(f"Training Progress (Epoch {epoch+1} Val Loss: {avg_val_loss})")

        # Optional: Save the best model based on validation loss
        # if avg_val_loss < best_val_loss:
        #      best_val_loss = avg_val_loss
        #      torch.save(model.state_dict(), 'best_model.pth') # Save model state dictionary

    print("\nTraining finished.")
    return model # Return the fully trained model

# %% [markdown]
# ## Evaluation Function
# Defines the `evaluate_model` function to calculate performance metrics (RMSE, MAE)
# on the validation set after training. This remains the same.

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
    model.eval() # Set model to evaluation mode (dropout off)

    y_true = [] # List to store true yield values
    y_pred = [] # List to store predicted yield values

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for x_val, y_val, fips_val in val_loader: # Unpack FIPS IDs
            x_val, y_val, fips_val = x_val.to(device), y_val.to(device), fips_val.to(device)
            outputs = model(x_val, fips_val) # Get predictions
            y_true.extend(y_val.cpu().numpy()) # Collect true values
            y_pred.extend(outputs.cpu().numpy()) # Collect predicted values

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
# Instantiate the dataset, prepare dataloaders using year-based splitting, determine model input dimensions,
# instantiate the model, and start the training process.

# %%
# --- Instantiate the Dataset ---
# Set the path to your data lake root and the target crop name
DATA_LAKE_ROOT = "data/data_lake_organized"
CROP_NAME = "corn" # Example crop
# Set the year(s) to hold out for testing/inference
HOLDOUT_YEAR = "2022" # Or ["2022", "2023"] if you have more recent data

dataset = None
try:
    dataset = CropYieldDataset(data_lake_dir=DATA_LAKE_ROOT, crop_name=CROP_NAME)
except FileNotFoundError as e:
    print(f"Fatal Error: {e}")
    print("Please ensure DATA_LAKE_ROOT path is correct and data exists.")

# --- Prepare DataLoaders using year-based splitting ---
train_loader, val_loader, holdout_dataset = None, None, None
if dataset is not None and len(dataset) > 0:
    # Get county mapping and number of unique counties from the dataset
    # These mappings are based on ALL data, including the holdout year, which is fine
    # as the model needs to know all possible counties.
    fips_to_id_mapping, id_to_fips_mapping = dataset.get_fips_mapping()
    num_fips = dataset.get_num_fips()
    print(f"Number of unique FIPS codes found across all data: {num_fips}")


    try:
        # Get dataloaders for training and validation, plus the holdout dataset
        train_loader, val_loader, holdout_dataset = get_dataloaders_by_year(
            dataset,
            holdout_year=HOLDOUT_YEAR,
            val_year_ratio=0.2, # Use 20% of *remaining* years for validation
            batch_size=32
        )

    except ValueError as e:
        print(f"Error creating dataloaders: {e}")
        # train_loader, val_loader, holdout_dataset remain None
    except TypeError as e:
        print(f"Error with holdout_year type: {e}")
        # train_loader, val_loader, holdout_dataset remain None


# --- Model Initialization and Training ---
trained_model = None
# Proceed only if training data loaders were successfully created
if train_loader is not None and val_loader is not None:
    # Detect input_dim (number of weather features) from the first sample
    # Accessing dataset[0] is safe here if train_loader/val_loader are not None,
    # as the base dataset must have samples.
    sample_weather, _, _ = dataset[0]
    input_dim = sample_weather.shape[-1]
    print(f"\nDetected input weather dimension: {input_dim}")


    # Instantiate the model
    model = LSTMTCNRegressor(
        input_dim=input_dim,
        num_fips=num_fips, # Use the total number of FIPS found in the dataset
        fips_embedding_dim=16, # Hyperparameter
        hidden_dim=64,        # Hyperparameter
        lstm_layers=1,        # Hyperparameter
        tcn_channels=[64, 32],# Hyperparameter
        dropout_rate=0.1      # Hyperparameter
    )

    print(f"\nModel architecture:\n{model}")


    # Train the model using only the train and validation loaders (from pre-holdout years)
    trained_model = train_model(model, train_loader, val_loader, num_epochs=20) # Revert to 20 epochs
    
    sample_weather, _, sample_fips = next(iter(train_loader))
    input_example = (sample_weather[:1], sample_fips[:1])  # Get a single example for MLflow logging
    save_model_to_mlflow(trained_model, input_example)
    
    # Evaluate the trained model on the validation set (pre-holdout years)
    evaluate_model(trained_model, val_loader)

else:
    print("\nSkipping model training and evaluation due to data loading or splitting issues.")
    # fips_to_id_mapping might not be defined here if dataset loading failed


# %% [markdown]
# ## Inference Function for Probabilistic Output
# Defines a function to perform inference on partial weather data up to a given date.
# It runs the trained model multiple times with dropout enabled to generate a sample
# of possible yield outcomes, which can then be visualized as a histogram.
# This remains the same, but we will now use it with the `holdout_dataset`.

# %%
def predict_distribution(
    model,
    data_lake_dir,
    year,
    county_fips,
    crop_name,
    inference_date, # Example: 'YYYY-MM-DD' (e.g., '2023-05-15')
    fips_to_id_mapping,
    num_samples=500, # Number of runs for dropout sampling (more samples give a better histogram)
    device=None
):
    """
    Predicts a yield distribution for a specific year, county, crop, and inference date.

    Loads weather data up to the inference_date, formats input, and runs the trained
    model multiple times with dropout enabled (Monte Carlo Dropout) to approximate
    the predictive distribution.

    Args:
        model (torch.nn.Module): The trained model (must be in evaluation mode initially).
        data_lake_dir (str): Path to the data lake root directory.
        year (int or str): The year for prediction (will be converted to str).
        county_fips (str): The FIPS code (string) for the county.
        crop_name (str): The name of the crop (used for messaging).
        inference_date (str): The date up to which weather data is available (YYYY-MM-DD).
        fips_to_id_mapping (dict): Dictionary mapping FIPS codes (str) to integer IDs,
                                   obtained from the training dataset (all data).
        num_samples (int): Number of forward passes with dropout to sample the distribution.
        device (torch.device, optional): The device (CPU/GPU) to use for inference.
                                         Defaults to automatically detecting GPU or using CPU.

    Returns:
        tuple: (predicted_yields_samples, bins, hist_counts) or (None, None, None) if error occurs.
               predicted_yields_samples (list): List of individual predicted yields from each sample run.
               bins (numpy.ndarray): Edges of the histogram bins.
               hist_counts (numpy.ndarray): Counts of samples falling into each bin.
    """
    year_str = str(year) # Ensure year is a string
    crop_name_lower = crop_name.lower()
    # Determine device
    device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # Ensure model is on the correct device

    print(f"\n--- Performing Inference ---")
    print(f"Inference Date: {inference_date}")
    print(f"County FIPS: {county_fips}")
    print(f"Crop: {crop_name}")
    print(f"Year: {year_str}")
    print(f"Number of samples for distribution: {num_samples}")
    print(f"Device: {device}")


    # --- 1. Load and Process Partial Weather Data ---
    fips_folder_path = Path(data_lake_dir) / county_fips
    year_folder_path = fips_folder_path / year_str
    weather_csv_path = year_folder_path / f"WeatherTimeSeries{year_str}.csv"

    if not weather_csv_path.exists():
        print(f"Error: Weather CSV not found at {weather_csv_path}")
        return None, None, None

    try:
        df_weather = pd.read_csv(weather_csv_path)
    except Exception as e:
        print(f"Error reading weather CSV {weather_csv_path}: {e}")
        return None, None, None


    # Filter weather data up to the specified inference date
    try:
        # Ensure date columns are present
        if not all(col in df_weather.columns for col in ['Year', 'Month', 'Day']):
             print(f"Error: Weather CSV for FIPS {county_fips}, Year {year_str} is missing Year, Month, or Day columns.")
             return None, None, None

        # Create a datetime column for filtering
        df_weather['Date'] = pd.to_datetime(df_weather[['Year', 'Month', 'Day']])
        inference_date_dt = pd.to_datetime(inference_date) # Convert inference date string to datetime object

        # Filter rows where the date is less than or equal to the inference date
        df_partial_weather = df_weather[df_weather['Date'] <= inference_date_dt].copy() # Use .copy()

        # Drop date-related columns and others not used as features
        cols_to_drop = ['Year', 'Month', 'Day', 'Date']
        existing_cols_to_drop = [col for col in cols_to_drop if col in df_partial_weather.columns]
        df_partial_weather = df_partial_weather.drop(columns=existing_cols_to_drop, errors='ignore')

        # Ensure all weather columns have numeric types and drop NaNs introduced by coerce
        for col in df_partial_weather.columns:
             df_partial_weather[col] = pd.to_numeric(df_partial_weather[col], errors='coerce')
        df_partial_weather.dropna(axis=1, how='all', inplace=True) # Drop columns all NaN

        # Drop rows with any NaN values after conversion
        initial_rows = len(df_partial_weather)
        df_partial_weather.dropna(axis=0, how='any', inplace=True)
        if len(df_partial_weather) < initial_rows:
             print(f"Warning: Dropped {initial_rows - len(df_partial_weather)} rows with NaN weather data up to {inference_date}.")
             if df_partial_weather.empty:
                  print(f"Error: No valid weather data rows left up to {inference_date}.")
                  return None, None, None


    except Exception as e:
        print(f"Error processing weather data for date filtering: {e}")
        return None, None, None

    if df_partial_weather.empty:
        print(f"Error: No valid weather data found up to {inference_date} for FIPS {county_fips}, year {year_str}")
        return None, None, None

    # Get the number of weather features expected by the model (from training dataset)
    # This assumes the order and number of features in the inference data matches training
    if model is None:
        print("Error: Model is not trained.")
        return None, None, None
    # Model's LSTM input size is weather_features + fips_embedding_dim
    expected_input_dim = model.lstm.input_size - model.fips_embedding.embedding_dim
    if df_partial_weather.shape[1] != expected_input_dim:
         print(f"Error: Weather data for inference has {df_partial_weather.shape[1]} features, but model expects {expected_input_dim}. Check column consistency between training data loading and inference loading.")
         # You might need to reorder/select/impute columns here if they aren't guaranteed to match the training columns
         return None, None, None


    # Convert the partial weather data to a PyTorch tensor
    # unsqueeze(0) adds the batch dimension (batch size = 1 for single inference sample)
    weather_tensor = torch.tensor(df_partial_weather.values, dtype=torch.float32).unsqueeze(0).to(device) # Shape: (1, T, features)


    # --- 2. Get County ID ---
    # Look up the integer ID for the target county FIPS code using the mapping from training
    if county_fips not in fips_to_id_mapping:
        print(f"Error: County FIPS {county_fips} not found in the training mapping ({len(fips_to_id_mapping)} counties trained).")
        print("The model was not trained on this specific county.")
        # Handle this case: could return None, predict a regional average, etc.
        return None, None, None

    fips_id = fips_to_id_mapping[county_fips]
    # Convert the single FIPS ID to a tensor with batch dimension
    fips_tensor = torch.tensor([fips_id], dtype=torch.long).to(device) # Shape: (1,)


    # --- 3. Run Model Multiple Times with Dropout (Monte Carlo Dropout) ---
    # Set model to training mode to enable dropout layers for sampling
    # This is a common technique for approximate uncertainty estimation (MC Dropout)
    model.train() # IMPORTANT: Enable dropout!

    predicted_yields = [] # List to collect predictions from each sample run
    with torch.no_grad(): # Ensure no gradients are computed during sampling
        # Run the model multiple times for the same input
        for i in tqdm.tqdm(range(num_samples), desc="Sampling Predictions"):
            # Pass weather and fips tensor (batch size 1)
            prediction = model(weather_tensor, fips_tensor)
            # .item() gets the scalar value from the 1-element tensor
            predicted_yields.append(prediction.item())

    model.eval() # Set model back to standard evaluation mode (dropout off)


    # --- 4. Generate Histogram Data ---
    # Calculate histogram bins and counts from the collected predictions
    if not predicted_yields: # Check if any predictions were collected
         print("Error: No predictions were generated.")
         return None, None, None

    # Use the range of predicted yields to define histogram bins
    min_yield = min(predicted_yields)
    max_yield = max(predicted_yields)

    # Handle edge case where all predictions are the same (e.g., if dropout=0 or num_samples=1)
    if min_yield == max_yield:
        print(f"Warning: All {num_samples} predictions are the same value: {min_yield:.2f}")
        # Create a single bin for the histogram
        # Create bounds slightly around the value to ensure the bar is visible
        bins = np.array([min_yield - 0.5, max_yield + 0.5])
        hist_counts = np.array([num_samples])
        print(f"Created single bin histogram: [{bins[0]:.2f}, {bins[1]:.2f}) with count {hist_counts[0]}")
    else:
        # Create bins - e.g., 20 bins between min and max predictions
        num_bins = 20 # Can be a parameter
        bins = np.linspace(min_yield, max_yield, num_bins + 1)
        # Calculate histogram counts. density=False gives counts per bin.
        hist_counts, bin_edges = np.histogram(predicted_yields, bins=bins, density=False)
        bins = bin_edges # Use the actual bin edges calculated by numpy

    print(f"Generated histogram data from {num_samples} samples.")
    print(f"Predicted Yield Range: [{min_yield:.2f}, {max_yield:.2f}]")
    print(f"Mean Predicted Yield: {np.mean(predicted_yields):.2f}")
    print(f"Std Dev of Predicted Yield: {np.std(predicted_yields):.2f}")


    # Return the raw samples, bin edges, and counts
    return predicted_yields, bins, hist_counts


# %% [markdown]
# ## Inference Execution
# Example of how to use the `predict_distribution` function with the trained model
# on the holdout year's data and plot the resulting histogram.

# %%
# --- How to use the inference function ---
# Ensure trained_model, fips_to_id_mapping, and holdout_dataset are available from the training block

# We perform inference if the model was trained and the necessary mappings/datasets exist.
if trained_model is not None and 'fips_to_id_mapping' in locals() and holdout_dataset is not None:

    print(f"\nAttempting inference for holdout year(s): {HOLDOUT_YEAR}")
    # We need to pick a specific sample from the holdout set for inference.
    # A holdout_dataset is a Subset of the original dataset.
    # holdout_dataset[0] will give us the first sample from the holdout set.
    # The sample format is (weather_tensor, yield_target, fips_id)
    # We also need the year and the *original* FIPS code string.
    # The year is stored in _samples_with_year in the original dataset,
    # and the FIPS ID mapping lets us get the string from the fips_id.

    if len(holdout_dataset) == 0:
        print(f"Cannot perform inference for holdout year(s) {HOLDOUT_YEAR} as the holdout dataset is empty.")
    else:
        # Pick the first sample from the holdout set as an example for inference
        # Access the original dataset's internal list to get the year string for the sample
        # This requires knowing the original index of the sample within the full dataset
        # holdout_dataset.indices contains the original indices
        sample_index_in_full_dataset = holdout_dataset.indices[0]
        _, true_yield_holdout, fips_id_holdout, year_str_holdout = dataset._samples_with_year[sample_index_in_full_dataset]

        # Get the FIPS code string from the ID
        county_fips_holdout = id_to_fips_mapping[fips_id_holdout]

        # Define the inference date (e.g., mid-season) for this sample's year
        # Replace with your desired inference date within the growing season of the holdout year
        # Example: Mid-June date
        inference_date_holdout = f"{year_str_holdout}-06-15" # Example date mid-season
        data_lake_root = DATA_LAKE_ROOT # Use the same data lake root as before
        target_crop = CROP_NAME # Use the same crop name as before
        inference_date_holdout = inference_date_holdout # Use the same inference date as before


        print(f"\n--- Example Inference on a Holdout Sample ---")
        print(f"Selected Holdout Sample: Year={year_str_holdout}, FIPS={county_fips_holdout}")
        print(f"True Yield for this sample (from dataset): {true_yield_holdout.item():.2f}")


        # Perform the probabilistic inference using the trained model
        predicted_yields_samples, bins, hist_counts = predict_distribution(
            trained_model,
            data_lake_root,
            year_str_holdout,      # The year of the holdout sample
            county_fips_holdout, # The FIPS code of the holdout sample
            target_crop,           # The crop name
            inference_date_holdout,# The date up to which weather is available
            fips_to_id_mapping,    # Pass the FIPS mapping
            num_samples=500,       # Number of MC Dropout samples
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # --- Visualize or Output Results ---
        if predicted_yields_samples is not None:
            # Plot the histogram
            plt.figure(figsize=(12, 7))
            # Use plt.bar with calculated bins/counts
            plt.bar(bins[:-1], hist_counts, width=np.diff(bins), edgecolor='black', alpha=0.7, align='edge')

            plt.xlabel(f"Predicted Yield ({target_crop.capitalize()} bushels)")
            plt.ylabel("Frequency (Count of Samples)")
            plt.title(f"Predicted Yield Distribution for {target_crop.capitalize()} in FIPS {county_fips_holdout}, Year {year_str_holdout}\n(Data up to {inference_date_holdout})")
            plt.grid(axis='y', alpha=0.5)


            # Print summary statistics
            print("\nInference Summary Statistics:")
            mean_pred = np.mean(predicted_yields_samples)
            median_pred = np.median(predicted_yields_samples)
            std_pred = np.std(predicted_yields_samples)
            print(f"  Mean Predicted Yield: {mean_pred:.2f}")
            print(f"  Median Predicted Yield: {median_pred:.2f}")
            print(f"  Standard Deviation: {std_pred:.2f}")
            print(f"  Min Predicted Yield: {min(predicted_yields_samples):.2f}")
            print(f"  Max Predicted Yield: {max(predicted_yields_samples):.2f}")
            print(f"  True Yield (Holdout Data): {true_yield_holdout.item():.2f}")


            # Add vertical lines for mean prediction and true yield
            plt.axvline(mean_pred, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean Pred: {mean_pred:.2f}')
            plt.axvline(true_yield_holdout.item(), color='green', linestyle='solid', linewidth=2, label=f'True Yield: {true_yield_holdout.item():.2f}')
            plt.legend()
            plt.tight_layout()
            plt.show() # Display the plot

            # Print histogram data points (optional)
            print("\nHistogram Data (Bin Range, Count):")
            # Ensure bin edges and counts match in length for iteration
            if len(bins) > 1 and len(hist_counts) == len(bins) -1:
                for i in range(len(hist_counts)):
                    print(f"  [{bins[i]:.2f} - {bins[i+1]:.2f}), Count: {hist_counts[i]}")
            elif len(bins) == 2 and len(hist_counts) == 1: # Case for single bin
                 print(f"  [{bins[0]:.2f} - {bins[1]:.2f}), Count: {hist_counts[0]}")
            else:
                 print("  Could not format histogram data points.")


        else:
            print("\nInference failed for the selected holdout sample.")

else:
    print("\nSkipping inference because model training was skipped or data components are missing.")

# %%