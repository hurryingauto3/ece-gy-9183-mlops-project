"""Custom exceptions for the model serving application."""

class ModelServingBaseError(Exception):
    """Base exception for this application."""
    pass

class FeatureServiceError(ModelServingBaseError):
    """Error related to the Feature Service interaction."""
    pass

class FeatureNotFoundError(FeatureServiceError, FileNotFoundError):
    """Specific error when features are not found via the Feature Service API."""
    # Inherits from FileNotFoundError for potential compatibility if needed,
    # but primarily signals a feature service issue.
    pass

class FeatureRequestError(FeatureServiceError):
    """Error during the request to the Feature Service (connection, timeout, etc.)."""
    pass

class FeatureResponseError(FeatureServiceError):
    """Error related to the response from the Feature Service (e.g., unexpected format, server error)."""
    pass

class ModelInferenceError(ModelServingBaseError):
    """Error related to the ML model inference process."""
    pass

class InvalidModelInputError(ModelInferenceError, ValueError):
    """Error when the input data provided to the model is invalid."""
    pass

class InvalidModelOutputError(ModelInferenceError, ValueError):
    """Error when the model produces an output in an unexpected format."""
    pass

class ConfigurationError(ModelServingBaseError):
    """Error related to application configuration."""
    pass
