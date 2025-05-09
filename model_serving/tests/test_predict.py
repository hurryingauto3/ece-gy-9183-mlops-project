import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, AsyncMock # Import AsyncMock

# Import the function to test
from model_serving.predict import predict_yield

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

@patch('model_serving.predict.get_features_from_api', new_callable=AsyncMock) # Mock the async API call function
async def test_predict_yield_success(mock_get_features_api):
    """
    Test the predict_yield function for a successful prediction using async API call.
    """
    # Arrange
    mock_model = MagicMock()
    mock_model.predict.return_value = [45.6]

    # Configure the mock get_features_from_api to return sample features
    sample_features = {'feature1': 10, 'feature2': 20}
    mock_get_features_api.return_value = sample_features

    test_county = "TestCounty"
    test_year = 2024

    # Act
    result = await predict_yield(mock_model, test_county, test_year) # Use await

    # Assert
    mock_get_features_api.assert_awaited_once_with(test_county, test_year) # Check awaited call
    mock_model.predict.assert_called_once()
    assert result == 45.6

@patch('model_serving.predict.get_features_from_api', new_callable=AsyncMock)
async def test_predict_yield_feature_not_found(mock_get_features_api):
    """
    Test predict_yield when get_features_from_api raises FileNotFoundError.
    """
    # Arrange
    mock_model = MagicMock()
    mock_get_features_api.side_effect = FileNotFoundError("Features not found via API")

    test_county = "MissingCounty"
    test_year = 2023

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Features not found via API"):
        await predict_yield(mock_model, test_county, test_year) # Use await

    mock_get_features_api.assert_awaited_once_with(test_county, test_year)
    mock_model.predict.assert_not_called()

@patch('model_serving.predict.get_features_from_api', new_callable=AsyncMock)
async def test_predict_yield_api_runtime_error(mock_get_features_api):
    """
    Test predict_yield when get_features_from_api raises RuntimeError (e.g., connection issue).
    """
    # Arrange
    mock_model = MagicMock()
    mock_get_features_api.side_effect = RuntimeError("Failed to connect to feature service")

    test_county = "ErrorCounty"
    test_year = 2022

    # Act & Assert
    with pytest.raises(RuntimeError, match="Failed to connect to feature service"):
        await predict_yield(mock_model, test_county, test_year) # Use await

    mock_get_features_api.assert_awaited_once_with(test_county, test_year)
    mock_model.predict.assert_not_called()


@patch('model_serving.predict.get_features_from_api', new_callable=AsyncMock)
async def test_predict_yield_invalid_feature_format(mock_get_features_api):
    """
    Test predict_yield when get_features_from_api returns an unexpected format.
    """
    # Arrange
    mock_model = MagicMock()
    # Simulate API returning something that cannot be parsed into a DataFrame easily
    mock_get_features_api.return_value = "this is not a dict or DataFrame json"

    test_county = "BadDataCounty"
    test_year = 2022

    # Act & Assert
    # The error might be ValueError depending on how pd.DataFrame or pd.read_json fails
    with pytest.raises(ValueError, match="Unexpected feature format received from API"):
        await predict_yield(mock_model, test_county, test_year) # Use await

    mock_get_features_api.assert_awaited_once_with(test_county, test_year)
    mock_model.predict.assert_not_called()
