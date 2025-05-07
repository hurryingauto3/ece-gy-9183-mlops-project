# model_training/utils.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn.utils.rnn # Needed for pad_sequence

import numpy as np
import tqdm # For progress bars
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random # Needed for random selection of years
import json # Needed for logging fips mapping artifact
import logging # Using standard logging for utils

# Import the dataset class - ensure correct relative import path
from .swift_data_loader import SwiftCropYieldDataset # Assuming swift_data_loader.py is in the same package

# Get logger for this module
logger = logging.getLogger(__name__)


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
    xs = [item[0] for item in batch]      # List of weather_tensors (varying lengths)
    ys = [item[1] for item in batch]      # List of yield_targets (scalar tensors)
    fips_ids = [item[2] for item in batch] # List of fips_ids (Python integers)

    # 1. Pad the sequences (weather tensors)
    # batch_first=True: (batch_size, max_seq_len, num_features)
    xs_padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0.0)

    # 2. Stack the targets
    ys_stacked = torch.stack(ys)

    # 3. Convert FIPS IDs to tensors and stack
    fips_ids_tensors = [torch.tensor(fips_id, dtype=torch.long) for fips_id in fips_ids]
    fips_ids_stacked = torch.stack(fips_ids_tensors)

    return xs_padded, ys_stacked, fips_ids_stacked


def get_dataloaders_by_year(dataset: SwiftCropYieldDataset, holdout_year: str | List[str] | None, val_year_ratio: float = 0.2, batch_size: int = 32):
    """
    Splits the dataset samples into training, validation, and optionally holdout sets based on year.
    Returns DataLoaders for training and validation, and the holdout dataset subset.

    Args:
        dataset (SwiftCropYieldDataset): The dataset instance.
        holdout_year (str | list[str] | None): The year(s) to hold out for testing/inference. Can be None,
                                             a single year (str), or a list of years (list of str).
        val_year_ratio (float): The proportion of *remaining* years (after holding out) to use for validation.
                                If < 1, it's a ratio. If >= 1, it's treated as the number of validation years.
        batch_size (int): The batch size for the DataLoaders.

    Returns:
        tuple: (train_loader, val_loader, holdout_dataset)
               train_loader (DataLoader): DataLoader for training data.
               val_loader (DataLoader): DataLoader for validation data.
               holdout_dataset (Subset | None): Subset dataset for holdout year(s), or None if no holdout year specified or found.

    Raises:
        ValueError: If the dataset is empty, holdout year not found among loaded samples, or not enough years for splitting.
        TypeError: If holdout_year is not None, a string, or list of strings/ints.
    """
    if len(dataset) == 0:
         raise ValueError("Dataset is empty. Cannot create dataloaders.")

    # Ensure holdout_year is a list of strings or None
    holdout_years = []
    if holdout_year is not None:
        if isinstance(holdout_year, int):
             holdout_years = [str(holdout_year)]
        elif isinstance(holdout_year, str):
             holdout_years = [holdout_year]
        elif isinstance(holdout_year, list):
             holdout_years = [str(y) for y in holdout_year]
        else:
             raise TypeError("holdout_year must be None, an integer, string, or a list of integers/strings.")


    all_years_in_dataset = dataset.get_years() # Get all unique years present in loaded samples
    samples_by_year_indices = dataset.get_sample_indices_by_year() # Get dict: year -> list of indices

    # Validate holdout year(s) against *loaded* years
    for year in holdout_years:
        if year not in all_years_in_dataset:
            # Note: This checks against years actually loaded from Swift, not just indexed.
            raise ValueError(f"Holdout year '{year}' not found in the loaded dataset samples. Available loaded years: {all_years_in_dataset}")

    # Separate holdout years and remaining years
    remaining_years = [year for year in all_years_in_dataset if year not in holdout_years]

    if not remaining_years:
        raise ValueError(f"No years left for training/validation after holding out {holdout_years}. Available loaded years: {all_years_in_dataset}")

    # Split remaining years into training and validation years
    random.seed(42) # Use a fixed seed for reproducible splitting if needed
    random.shuffle(remaining_years) # Shuffle years before splitting

    if val_year_ratio >= 1: # Treat ratio as a fixed number of years
        num_val_years = int(val_year_ratio) # Use this number of years for validation
        if num_val_years < 1: # Ensure at least 1 validation year if possible
             logger.warning(f"Specified val_year_ratio ({val_year_ratio}) results in less than 1 validation year. Setting to 1 if possible.")
             num_val_years = 1

        if num_val_years >= len(remaining_years):
             raise ValueError(f"Number of validation years ({num_val_years}) is >= remaining years ({len(remaining_years)}). Cannot create training set. Remaining years: {remaining_years}")

        val_years = remaining_years[:num_val_years]
        train_years = remaining_years[num_val_years:]
    else: # Treat ratio as a proportion
        num_val_years = max(1, int(len(remaining_years) * val_year_ratio)) # Ensure at least 1 val year if possible
        if len(remaining_years) - num_val_years < 1 and len(remaining_years) >= 2:
             # If ratio makes train set empty, maybe take one less val year if possible
             # Only do this if there are at least 2 remaining years total
             logger.warning(f"Validation ratio {val_year_ratio} leaves no years for training ({len(remaining_years)-num_val_years} years). Reducing validation years by 1.")
             num_val_years -= 1
             if num_val_years < 1:
                   raise ValueError(f"Cannot split years. Remaining years ({len(remaining_years)}) too few for training/validation with ratio {val_year_ratio} and holdout {holdout_years}.")

        val_years = remaining_years[:num_val_years]
        train_years = remaining_years[num_val_years:]

    if not train_years:
         raise ValueError(f"Training set years is empty after splitting. Remaining years: {remaining_years}, Val years: {val_years}")
    if not val_years:
         raise ValueError(f"Validation set years is empty after splitting. Remaining years: {remaining_years}, Train years: {train_years}")


    logger.info(f"Splitting data by year:")
    logger.info(f"  Holdout year(s): {holdout_years if holdout_years else 'None'}")
    logger.info(f"  Training year(s): {sorted(train_years)}")
    logger.info(f"  Validation year(s): {sorted(val_years)}")

    # Collect sample indices for each split
    train_indices = [idx for year in train_years for idx in samples_by_year_indices.get(year, [])]
    val_indices = [idx for year in val_years for idx in samples_by_year_indices.get(year, [])]
    holdout_indices = [idx for year in holdout_years for idx in samples_by_year_indices.get(year, [])]


    if not train_indices:
         raise ValueError(f"Training set is empty after collecting indices for years {train_years}.")
    if not val_indices:
         raise ValueError(f"Validation set is empty after collecting indices for years {val_years}.")

    holdout_dataset = None
    if holdout_indices:
         holdout_dataset = Subset(dataset, holdout_indices)
         logger.info(f"Holdout samples: {len(holdout_dataset)}")
    else:
         logger.warning(f"Holdout set for year(s) {holdout_years} is empty or None specified.")


    # Create Subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)


    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")


    # Create DataLoaders for train and validation using the custom collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    logger.info(f"Training DataLoader created with {len(train_loader)} batches.")
    logger.info(f"Validation DataLoader created with {len(val_loader)} batches.")

    return train_loader, val_loader, holdout_dataset


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 30, lr: float = 1e-3):
    """
    Trains the provided model using the training and validation dataloaders.
    Logs epoch metrics to MLflow.

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
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # best_val_loss = float('inf') # Keep if implementing early stopping

    logger.info(f"Starting model training on {device} for {num_epochs} epochs...")

    # Import MLflow here to avoid import issues if called outside MLflow run context
    import mlflow
    import mlflow.pytorch


    for epoch in tqdm.tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        epoch_loss = 0

        for x_batch, y_batch, fips_batch in train_loader:
            x_batch, y_batch, fips_batch = x_batch.to(device), y_batch.to(device), fips_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(x_batch, fips_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0) # Accumulate sum of losses over batch samples
        avg_train_loss = epoch_loss / len(train_loader.dataset) # Average loss over total samples in train set


        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val, fips_val in val_loader:
                x_val, y_val, fips_val = x_val.to(device), y_val.to(device), fips_val.to(device)
                y_pred = model(x_val, fips_val)
                val_loss += criterion(y_pred, y_val).item() * x_val.size(0) # Accumulate sum of losses over batch samples
        avg_val_loss = val_loss / len(val_loader.dataset) # Average loss over total samples in val set


        # Log metrics to MLflow for this epoch
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        # Log to console/logs
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")


    logger.info("Model training finished.")
    return model


def evaluate_model(model: nn.Module, data_loader: DataLoader):
    """
    Evaluates the model's performance on a given dataset (validation or holdout).
    Logs final metrics to MLflow.

    Args:
        model (torch.nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the evaluation data.

    Returns:
        tuple: (rmse, mae) - Root Mean Squared Error and Mean Absolute Error.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    y_true = []
    y_pred = []

    logger.info(f"Starting model evaluation on {device}...")

    # Use tqdm for progress bar during evaluation if data_loader is large
    with torch.no_grad():
        for x_batch, y_batch, fips_batch in tqdm.tqdm(data_loader, desc="Evaluating"):
            x_batch, y_batch, fips_batch = x_batch.to(device), y_batch.to(device), fips_batch.to(device)
            outputs = model(x_batch, fips_batch)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    # Calculate RMSE and MAE using scikit-learn
    rmse = mean_squared_error(y_true, y_pred, squared=False) # Calculate RMSE correctly
    mae = mean_absolute_error(y_true, y_pred)


    logger.info(f"Evaluation complete.")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE:  {mae:.4f}")

    # Log final evaluation metrics to MLflow
    # Ensure these metric names are distinct if evaluating both val and holdout in train_job
    # Example: "val_rmse", "val_mae", "holdout_rmse", "holdout_mae"
    # The calling train_job script should log these based on *which* loader was evaluated.
    # Returning them for the caller (train_job.py) to log is more flexible.
    # MLflow logging moved to train_job.py

    return rmse, mae

# The predict_distribution and Inference Execution sections from your notebook
# should NOT be included in the train_job.py script. Probabilistic inference
# is a separate task. Keep them in the notebook for interactive analysis.
# If you need probabilistic inference from the serving API, that logic would
# go into model_serving/predict.py (adapted to read data from Feature Service
# up to an inference date and use model.train() with multiple runs).