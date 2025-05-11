from pydantic import BaseModel, Field
from typing import List, Dict, Any

# This model represents a single day's weather features
# We use Dict[str, Any] as the keys (feature names) are dynamic
DailyWeatherFeatures = Dict[str, Any]

class FeaturesResponse(BaseModel):
    """Response model for the /features endpoint."""
    fips_code: str = Field(..., description="FIPS code of the county.")
    year: int = Field(..., description="Year of the weather data.")
    weather_data: List[DailyWeatherFeatures] = Field(..., description="List of daily weather feature dictionaries for the growing season.")
    # Add metadata like start/end date of the sequence if useful in the future