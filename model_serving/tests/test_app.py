import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from httpx import Response, RequestError  # Import RequestError

from model_serving.app import app  # Import app directly
from model_serving.mlflow_loader import get_model  # Import the dependency function
from model_serving.exceptions import ModelServingBaseError  # Import base error

# Mark all tests in this module as asyncio (needed for async TestClient calls)
pytestmark = pytest.mark.asyncio

# Define the base URL for the mocked feature service
MOCK_FEATURE_SERVICE_BASE_URL = "http://mock-features.com/"  # Use base URL with trailing slash

@pytest.fixture(scope="module")
def mock_model_fixture():
    """Provides a reusable mock model instance."""
    model = MagicMock()
    model.predict.return_value = [45.6]  # Default mock prediction for single
    return model

@pytest.fixture(scope="module")
def client(mock_model_fixture):
    """Create a TestClient instance with dependency overrides."""
    # Override the get_model dependency to return the mock model
    app.dependency_overrides[get_model] = lambda: mock_model_fixture

    with patch('model_serving.mlflow_loader.settings') as mock_settings:
        # Configure mock settings
        mock_settings.feature_service.url = "http://mock-features.com"
        mock_settings.mlflow.tracking_uri = "mock_mlflow_uri"
        mock_settings.mlflow.model_name = "MockModel"
        mock_settings.mlflow.model_stage = "Testing"
        test_client = TestClient(app)
        yield test_client

    # Clean up overrides after tests
    app.dependency_overrides = {}

# === Test /health endpoint ===
async def test_health_check_ok(client: TestClient, httpx_mock, mock_model_fixture):
    """Test successful health check when all dependencies are ok."""
    # Mock the HEAD request to the feature service base URL
    httpx_mock.add_response(
        url=MOCK_FEATURE_SERVICE_BASE_URL,
        method="HEAD",
        status_code=200  # Simulate successful connection
    )

    response = await client.get("/health")  # Use await for async endpoint

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

async def test_health_check_model_fail(client: TestClient, httpx_mock):
    """Test health check failure when the model dependency fails."""
    # Override dependency to simulate model loading failure for this test
    def fail_get_model():
        raise ModelServingBaseError("Simulated model load failure")
    app.dependency_overrides[get_model] = fail_get_model

    # Mock feature service just in case
    httpx_mock.add_response(url=MOCK_FEATURE_SERVICE_BASE_URL, method="HEAD", status_code=200)

    response = await client.get("/health")
    # Expect 503 due to the custom exception handler for ModelServingBaseError
    assert response.status_code == 503
    assert "Simulated model load failure" in response.json()["detail"]

    app.dependency_overrides = {}  # Clean up override for this specific test

async def test_health_check_feature_service_fail_connect(client: TestClient, httpx_mock, mock_model_fixture):
    """Test health check failure when feature service connection fails."""
    # Ensure model dependency is ok for this test
    app.dependency_overrides[get_model] = lambda: mock_model_fixture
    # Mock the HEAD request to raise a connection error
    httpx_mock.add_exception(RequestError("Connection failed"), url=MOCK_FEATURE_SERVICE_BASE_URL, method="HEAD")

    response = await client.get("/health")

    assert response.status_code == 503
    assert "Cannot reach Feature Service" in response.json()["detail"]
    app.dependency_overrides = {}  # Clean up

async def test_health_check_feature_service_fail_500(client: TestClient, httpx_mock, mock_model_fixture):
    """Test health check failure when feature service returns 5xx."""
    # Ensure model dependency is ok
    app.dependency_overrides[get_model] = lambda: mock_model_fixture
    # Mock the HEAD request to return a 500 status
    httpx_mock.add_response(
        url=MOCK_FEATURE_SERVICE_BASE_URL,
        method="HEAD",
        status_code=500  # Simulate server error from feature service
    )

    response = await client.get("/health")

    assert response.status_code == 503
    assert "Cannot reach Feature Service" in response.json()["detail"]
    app.dependency_overrides = {}  # Clean up

async def test_health_check_feature_service_ok_4xx(client: TestClient, httpx_mock, mock_model_fixture):
    """Test health check success even if feature service returns 4xx (e.g., 404 on base URL)."""
    # Ensure model dependency is ok
    app.dependency_overrides[get_model] = lambda: mock_model_fixture
    # Mock the HEAD request to return a 404 status (still reachable)
    httpx_mock.add_response(
        url=MOCK_FEATURE_SERVICE_BASE_URL,
        method="HEAD",
        status_code=404  # Simulate client error (e.g., base URL not found, but service is up)
    )

    response = await client.get("/health")

    # Health check should still pass as the service is reachable
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    app.dependency_overrides = {}  # Clean up

# === Test /model_info endpoint ===
def test_model_info_ok(client: TestClient):
    response = client.get("/model_info")
    assert response.status_code == 200
    # Assert based on the patched settings values
    assert response.json() == {
        "model_name": "MockModel",
        "model_stage": "Testing",
        "mlflow_uri": "mock_mlflow_uri"
    }

# === Test /predict endpoint ===
async def test_predict_success(client: TestClient, httpx_mock, mock_model_fixture):
    """Test successful single prediction."""
    # Mock the feature service API call
    httpx_mock.add_response(
        url=f"{MOCK_FEATURE_SERVICE_BASE_URL}features?county=TestCounty&year=2023",
        method="GET",
        json={"feature1": 10, "feature2": 20}  # Sample feature response
    )
    
    # Reset mock model predict return value if needed for this specific test
    mock_model_fixture.predict.reset_mock()
    mock_model_fixture.predict.return_value = [55.5] 

    request_data = {"county": "TestCounty", "year": 2023}
    response = await client.post("/predict", json=request_data)  # Use await for async client call

    assert response.status_code == 200
    assert response.json() == {
        "county": "TestCounty",
        "year": 2023,
        "predicted_yield": 55.5  # Matches the mocked model output
    }
    mock_model_fixture.predict.assert_called_once()  # Check model was called

async def test_predict_feature_not_found(client: TestClient, httpx_mock):
    """Test prediction when feature service returns 404."""
    httpx_mock.add_response(
        url=f"{MOCK_FEATURE_SERVICE_BASE_URL}features?county=MissingCounty&year=2023",
        method="GET",
        status_code=404
    )

    request_data = {"county": "MissingCounty", "year": 2023}
    response = await client.post("/predict", json=request_data)

    assert response.status_code == 404
    assert "Features not found via API" in response.json()["detail"]

async def test_predict_feature_service_error(client: TestClient, httpx_mock):
    """Test prediction when feature service returns 500."""
    httpx_mock.add_response(
        url=f"{MOCK_FEATURE_SERVICE_BASE_URL}features?county=ErrorCounty&year=2023",
        method="GET",
        status_code=500
    )

    request_data = {"county": "ErrorCounty", "year": 2023}
    response = await client.post("/predict", json=request_data)

    assert response.status_code == 503  # We map runtime errors to 503
    assert "Failed to fetch features from API" in response.json()["detail"]

async def test_predict_invalid_input_year(client: TestClient):
    """Test prediction with invalid input data (year out of range)."""
    request_data = {"county": "ValidCounty", "year": 1776}  # Invalid year
    response = await client.post("/predict", json=request_data)

    assert response.status_code == 422  # FastAPI validation error

async def test_predict_invalid_input_county(client: TestClient):
    """Test prediction with invalid input data (empty county)."""
    request_data = {"county": " ", "year": 2023}  # Invalid county
    response = await client.post("/predict", json=request_data)

    assert response.status_code == 422  # FastAPI validation error

# === Test /predict_batch endpoint ===
async def test_predict_batch_success(client: TestClient, httpx_mock, mock_model_fixture):
    """Test successful batch prediction."""
    # Mock feature service calls for each item in the batch
    httpx_mock.add_response(
        url=f"{MOCK_FEATURE_SERVICE_BASE_URL}features?county=CountyA&year=2022",
        method="GET", json={"feature1": 1, "feature2": 2}
    )
    httpx_mock.add_response(
        url=f"{MOCK_FEATURE_SERVICE_BASE_URL}features?county=CountyB&year=2023",
        method="GET", json={"feature1": 3, "feature2": 4}
    )

    # Mock the model's batch prediction output
    # IMPORTANT: Ensure the order matches the order of valid_features in predict_yield_batch
    mock_model_fixture.predict.reset_mock()
    mock_model_fixture.predict.return_value = [11.1, 22.2]  # Predictions for CountyA, CountyB

    request_data = {
        "requests": [
            {"county": "CountyA", "year": 2022},
            {"county": "CountyB", "year": 2023}
        ]
    }
    response = await client.post("/predict_batch", json=request_data)

    assert response.status_code == 200
    assert response.json() == {
        "responses": [
            {"county": "CountyA", "year": 2022, "predicted_yield": 11.1, "error": None},
            {"county": "CountyB", "year": 2023, "predicted_yield": 22.2, "error": None}
        ]
    }
    mock_model_fixture.predict.assert_called_once()  # Model predict called once for the batch

async def test_predict_batch_partial_failure(client: TestClient, httpx_mock, mock_model_fixture):
    """Test batch prediction with one item failing feature fetch."""
    # Mock successful feature fetch for CountyA
    httpx_mock.add_response(
        url=f"{MOCK_FEATURE_SERVICE_BASE_URL}features?county=CountyA&year=2022",
        method="GET", json={"feature1": 1, "feature2": 2}
    )
    # Mock failed feature fetch (404) for CountyC
    httpx_mock.add_response(
        url=f"{MOCK_FEATURE_SERVICE_BASE_URL}features?county=CountyC&year=2024",
        method="GET", status_code=404
    )

    # Mock model prediction for the single successful item (CountyA)
    mock_model_fixture.predict.reset_mock()
    mock_model_fixture.predict.return_value = [33.3]

    request_data = {
        "requests": [
            {"county": "CountyA", "year": 2022},
            {"county": "CountyC", "year": 2024}
        ]
    }
    response = await client.post("/predict_batch", json=request_data)

    assert response.status_code == 200
    response_data = response.json()["responses"]
    assert len(response_data) == 2
    # Check CountyA (success)
    assert response_data[0]["county"] == "CountyA"
    assert response_data[0]["predicted_yield"] == 33.3
    assert response_data[0]["error"] is None
    # Check CountyC (failure)
    assert response_data[1]["county"] == "CountyC"
    assert response_data[1]["predicted_yield"] is None
    assert "Features not found via API" in response_data[1]["error"]

    mock_model_fixture.predict.assert_called_once()  # Model still called once for the valid part of the batch

async def test_predict_batch_empty_request(client: TestClient):
    """Test batch prediction with an empty request list."""
    request_data = {"requests": []}
    response = await client.post("/predict_batch", json=request_data)
    assert response.status_code == 422  # Validation error (min_length=1)

async def test_predict_batch_invalid_item(client: TestClient):
    """Test batch prediction with an invalid item in the list."""
    request_data = {
        "requests": [
            {"county": "CountyA", "year": 2022},
            {"county": "CountyB"}  # Missing year
        ]
    }
    response = await client.post("/predict_batch", json=request_data)
    assert response.status_code == 422  # Validation error
