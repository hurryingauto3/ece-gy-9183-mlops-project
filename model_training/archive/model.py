from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class FeaturesResponseItem(BaseModel):
    # Assuming keys are weather feature names and values are floats/ints
    # This model represents a single day's features
    pass  # Define feature fields explicitly later if needed, or use Dict[str, Any]


class FeaturesResponse(BaseModel):
    fips_code: str
    year: int
    weather_data: List[Dict[str, Any]] = Field(
        ..., description="List of daily weather feature dictionaries for the season."
    )
    # Add metadata like start/end date of the sequence if useful


# If implementing mid-season inference
class FeaturesRequest(BaseModel):
    county: str
    year: int
    inference_date: Optional[str] = Field(
        None, description="Optional date (YYYY-MM-DD) to limit weather data."
    )
