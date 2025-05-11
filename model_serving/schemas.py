from pydantic import BaseModel, field_validator, Field
from typing import List, Dict, Any, Optional
from datetime import date

# --- Pydantic Models for API Requests and Responses ---

class PredictionRequest(BaseModel):
    county: str = Field(..., example="TestCounty", description="Name of the county (often FIPS code).")
    year: int = Field(
        ..., example=2023, description="Year for the prediction (e.g., 1980-2050)."
    )
    cut_off_date: date = Field(..., example="2023-07-15", description="Cut-off date for features (YYYY-MM-DD).")
    crop: str = Field(..., example="corn", description="Type of crop for the prediction.")
    histogram_bins: List[float] = Field(
        ..., 
        example=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        description="List of bin edges for the output histogram."
    )

    @field_validator("year")
    def year_must_be_reasonable(cls, v): # Added cls for Pydantic v2 validator
        if not (1980 <= v <= 2050):  # Example range, adjust as needed
            raise ValueError("Year must be between 1980 and 2050")
        return v

    @field_validator("county")
    def county_must_not_be_empty(cls, v): # Added cls
        if not v or v.isspace():
            raise ValueError("County cannot be empty")
        return v.strip()

    @field_validator("histogram_bins")
    def histogram_bins_must_be_valid(cls, v): # Added cls
        if len(v) < 2:
            raise ValueError("histogram_bins must contain at least two edges.")
        if not all(v[i] < v[i+1] for i in range(len(v)-1)):
            raise ValueError("histogram_bins edges must be strictly increasing.")
        return v


class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest] = Field(
        ..., min_length=1, description="A list of prediction requests."
    )


class BatchPredictionResponseItem(BaseModel):
    county: str = Field(
        ..., example="TestCounty", description="Name of the county from the request."
    )
    year: int = Field(..., example=2023, description="Year from the request.")
    cut_off_date: date = Field(..., example="2023-07-15", description="Cut-off date from the request.")
    crop: str = Field(..., example="corn", description="Crop type from the request.")
    predicted_histogram: Optional[Dict[str, List[float]]] = Field(
        None, 
        example={"bin_edges": [0, 10, 20], "probabilities": [0.2, 0.8]},
        description="Predicted histogram (bin_edges and probabilities/counts), if successful."
    )
    error: Optional[str] = Field(
        None,
        example="Features not found",
        description="Error message, if prediction failed for this item.",
    )


class BatchPredictionResponse(BaseModel):
    responses: List[BatchPredictionResponseItem] = Field(
        ..., description="List of results corresponding to the batch requests."
    )


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")


class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., example="AgriYieldPredictor")
    model_stage: str = Field(..., example="Production")
    mlflow_uri: str # Changed from HttpUrl to str to avoid HttpUrl import here, can be validated in app.py if needed
    # model_version: str = Field(..., example="1") 