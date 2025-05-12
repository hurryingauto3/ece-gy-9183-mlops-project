import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

def process_prediction_data(raw_data: Dict[str, List[int]]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
    """
    Process the raw prediction data from the API into a format suitable for visualization.
    
    Args:
        raw_data: Dictionary mapping crop types to lists of prediction values
        
    Returns:
        A tuple containing:
        - Dictionary mapping crop types to DataFrames with processed prediction data
        - Dictionary mapping crop types to dictionaries with summary statistics
    """
    try:
        processed_dfs = {}
        summary_stats = {}
        
        for crop_type, yield_values in raw_data.items():
            # Create a DataFrame for this crop
            df = pd.DataFrame({"yield_value": yield_values})
            
            # Calculate summary statistics for this crop
            crop_stats = {
                "mean": df["yield_value"].mean(),
                "median": df["yield_value"].median(),
                "min": df["yield_value"].min(),
                "max": df["yield_value"].max(),
                "std": df["yield_value"].std()
            }
            
            processed_dfs[crop_type] = df
            summary_stats[crop_type] = crop_stats
        
        return processed_dfs, summary_stats
        
    except Exception as e:
        # Handle any processing errors
        raise Exception(f"Error processing prediction data: {str(e)}")

def process_historical_data(historical_data: Dict[str, Dict[int, float]], prediction_year: int = 0) -> pd.DataFrame:
    """
    Process historical crop yield data into a format suitable for visualization.
    
    Args:
        historical_data: Dictionary mapping crop types to dictionaries of year->yield value
        prediction_year: The year for which we have predictions (to mark in visuals)
        
    Returns:
        DataFrame with processed historical data for plotting time series
    """
    try:
        # Create a list to hold all data records
        records = []
        
        for crop_type, year_data in historical_data.items():
            for year, yield_value in year_data.items():
                records.append({
                    "crop_type": crop_type,
                    "year": year,
                    "yield_value": yield_value,
                    "data_type": "Historical"
                })
        
        # Create a DataFrame from the records
        df = pd.DataFrame(records)
        
        return df
        
    except Exception as e:
        # Handle any processing errors
        raise Exception(f"Error processing historical data: {str(e)}")

def combine_historical_and_prediction(
    historical_df: pd.DataFrame, 
    prediction_dfs: Dict[str, pd.DataFrame], 
    prediction_year: int
) -> pd.DataFrame:
    """
    Combine historical data with prediction data for visualization.
    
    Args:
        historical_df: DataFrame with historical yield data
        prediction_dfs: Dictionary mapping crop types to DataFrames with prediction data
        prediction_year: The year for which we have predictions
        
    Returns:
        DataFrame with combined historical and prediction data
    """
    try:
        # Create a list to hold records for the prediction data
        prediction_records = []
        
        for crop_type, pred_df in prediction_dfs.items():
            # Get the mean prediction value for each crop
            mean_prediction = pred_df["yield_value"].mean()
            
            # Add a record for the prediction
            prediction_records.append({
                "crop_type": crop_type,
                "year": prediction_year,
                "yield_value": mean_prediction,
                "data_type": "Prediction"
            })
        
        # Create a DataFrame from the prediction records
        prediction_df = pd.DataFrame(prediction_records)
        
        # Combine historical and prediction DataFrames
        combined_df = pd.concat([historical_df, prediction_df], ignore_index=True)
        
        return combined_df
        
    except Exception as e:
        # Handle any processing errors
        raise Exception(f"Error combining historical and prediction data: {str(e)}")

# --- NEW FUNCTION for histogram data ---
def process_histogram_prediction(api_response: Dict[str, Any]) -> pd.DataFrame:
    """
    Processes the histogram prediction from the API into a DataFrame for plotting.

    Args:
        api_response: The parsed JSON response from the /predict endpoint.
                      Expected to have a 'predicted_histogram' key containing
                      'bin_edges' and 'probabilities'.

    Returns:
        A Pandas DataFrame with columns 'Bin Range' and 'Probability',
        suitable for st.bar_chart.
        Returns an empty DataFrame if the expected data is not present.
    """
    if not api_response or "predicted_histogram" not in api_response:
        # st.error("Prediction data is missing 'predicted_histogram'.") # Cannot use st here
        print("Data Processor Error: Prediction data is missing 'predicted_histogram'.")
        return pd.DataFrame(columns=['Bin Range', 'Probability'])

    histogram_data = api_response["predicted_histogram"]
    if not histogram_data or "bin_edges" not in histogram_data or "probabilities" not in histogram_data:
        print("Data Processor Error: Histogram data is missing 'bin_edges' or 'probabilities'.")
        return pd.DataFrame(columns=['Bin Range', 'Probability'])

    bin_edges = histogram_data["bin_edges"]
    probabilities = histogram_data["probabilities"]

    if len(bin_edges) != len(probabilities) + 1:
        print("Data Processor Error: Length of bin_edges must be one more than probabilities.")
        # Attempt to reconcile if possible, e.g. if probabilities represent midpoints or counts for N-1 bins
        # For now, return empty if strictly N and N+1 bins/probabilities are expected by this processor logic
        return pd.DataFrame(columns=['Bin Range', 'Probability'])
    
    if not probabilities: # Handle empty probabilities list
        print("Data Processor Warning: Probabilities list is empty.")
        return pd.DataFrame(columns=['Bin Range', 'Probability'])

    bin_labels = []
    for i in range(len(probabilities)):
        bin_labels.append(f"{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}")
    
    chart_df = pd.DataFrame({
        'Bin Range': bin_labels,
        'Probability': probabilities
    })
    return chart_df
