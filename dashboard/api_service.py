import requests
import time
import json
from datetime import date
from typing import Dict, Any, List, Union, Tuple
import os # Added to get environment variable

def fetch_prediction_data(
    county: str,
    year: int,
    crop_types: List[str] = ["Corn", "Wheat", "Soybeans", "Rice"]
) -> Dict[str, List[int]]:
    """
    Fetch crop yield prediction data from the API.
    
    Args:
        county: The county to get predictions for
        year: The year for prediction
        crop_types: List of crop types to get predictions for (default: top 4 crops)
    
    Returns:
        A dictionary mapping crop types to lists of prediction values
    
    Raises:
        Exception: If there's an error fetching the data
    """
    try:
        # API endpoint - this should be replaced with the actual API endpoint
        api_url = "https://api.example.com/crop-predictions"
        
        # If no crop types provided, use default top 4
        if not crop_types:
            crop_types = ["Corn", "Wheat", "Soybeans", "Rice"]
        
        # Prepare request parameters
        params = {
            "county": county,
            "year": year,
            "crop_types": ",".join(crop_types)
        }
        
        # Add some delay to simulate real API call
        time.sleep(1)
        
        # For demonstration purposes, we'll simulate the API response
        # In a real application, this would be an actual API call:
        # response = requests.get(api_url, params=params)
        # response.raise_for_status()  # Raise an exception for HTTP errors
        # prediction_data = response.json()
        
        # Simulate API response with random data
        # In a real app, this would come from the actual API
        import random
        
        prediction_data = {}
        for crop in crop_types:
            # Different crops have different baseline yields
            base_yields = {
                "Corn": random.randint(150, 200),
                "Wheat": random.randint(50, 80),
                "Soybeans": random.randint(40, 60),
                "Rice": random.randint(80, 120),
                "Cotton": random.randint(800, 1200),  # in lbs per acre
                "Barley": random.randint(60, 100)
            }
            
            base_yield = base_yields.get(crop, random.randint(30, 70))
            variation = random.randint(5, 15)
            
            # Simulate the JSON response that would come from the API
            prediction_data[crop] = [
                max(0, int(random.gauss(base_yield, variation))) for _ in range(100)
            ]
        
        return prediction_data
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse API response: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")

def fetch_historical_data(
    county: str,
    start_year: int,
    end_year: int,
    crop_types: List[str] = ["Corn", "Wheat", "Soybeans", "Rice"]
) -> Dict[str, Dict[int, float]]:
    """
    Fetch historical crop yield data from the API.
    
    Args:
        county: The county to get data for
        start_year: The first year in the historical range
        end_year: The last year in the historical range
        crop_types: List of crop types to get data for (default: top 4 crops)
    
    Returns:
        A dictionary mapping crop types to dictionaries of year->yield value
    
    Raises:
        Exception: If there's an error fetching the data
    """
    try:
        # API endpoint - this should be replaced with the actual API endpoint
        api_url = "https://api.example.com/historical-yields"
        
        # If no crop types provided, use default top 4
        if not crop_types:
            crop_types = ["Corn", "Wheat", "Soybeans", "Rice"]
        
        # Prepare request parameters
        params = {
            "county": county,
            "start_year": start_year,
            "end_year": end_year,
            "crop_types": ",".join(crop_types)
        }
        
        # Add some delay to simulate real API call
        time.sleep(1)
        
        # Simulate API response with semi-realistic data
        import random
        
        historical_data = {}
        for crop in crop_types:
            # Different crops have different baseline yields
            base_yields = {
                "Corn": 175,
                "Wheat": 65,
                "Soybeans": 50,
                "Rice": 100,
                "Cotton": 1000,  # in lbs per acre
                "Barley": 80
            }
            
            base_yield = base_yields.get(crop, 50)
            trend = random.uniform(0.5, 1.5)  # Slight upward trend over time
            
            # Create year-to-yield mapping with a realistic trend
            crop_data = {}
            for year in range(start_year, end_year + 1):
                # Add some annual variation plus a slight upward trend
                year_factor = 1.0 + (year - start_year) * (trend / 100)
                annual_variation = random.uniform(0.85, 1.15)  # 15% variation year to year
                crop_data[year] = round(base_yield * year_factor * annual_variation, 1)
            
            historical_data[crop] = crop_data
        
        return historical_data
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse API response: {str(e)}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")

def fetch_single_crop_histogram_prediction(
    county_fips: str,
    cut_off_date: date,
    crop_name: str,
    histogram_bins: List[float]
) -> Dict[str, Any]:
    """
    Fetch a single crop yield histogram prediction from the model-serving API.

    Args:
        county_fips: The FIPS code of the county.
        cut_off_date: The cut-off date for the prediction.
        crop_name: The name of the crop.
        histogram_bins: A list of floats defining the bin edges for the histogram.

    Returns:
        A dictionary containing the API response (parsed JSON).
    
    Raises:
        Exception: If there's an error fetching or parsing the data.
    """
    api_base_url = os.environ.get("MODEL_SERVING_API_URL", "http://model-serving:8000")
    predict_url = f"{api_base_url}/predict"

    payload = {
        "county": county_fips,
        "year": cut_off_date.year, # API might use this, or it might just use cut_off_date
        "cut_off_date": cut_off_date.isoformat(),
        "crop": crop_name,
        "histogram_bins": histogram_bins
    }

    try:
        print(f"Dashboard: Sending request to {predict_url} with payload: {payload}") # Debug print
        response = requests.post(predict_url, json=payload, timeout=20) # Increased timeout
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        prediction_data = response.json()
        print(f"Dashboard: Received response: {prediction_data}") # Debug print
        return prediction_data
    except requests.exceptions.Timeout:
        raise Exception(f"API request timed out after 20 seconds: {predict_url}")
    except requests.exceptions.HTTPError as http_err:
        error_content = response.text # Or response.json() if the error is JSON
        raise Exception(f"API request failed with HTTP error {http_err.response.status_code} for {predict_url}: {error_content}")
    except requests.exceptions.RequestException as req_err:
        raise Exception(f"API request failed for {predict_url}: {req_err}")
    except json.JSONDecodeError:
        raise Exception(f"Failed to parse API response from {predict_url} as JSON. Response text: {response.text}")
    except Exception as e: # Catch any other unexpected errors
        raise Exception(f"An unexpected error occurred while calling prediction API at {predict_url}: {e}")
