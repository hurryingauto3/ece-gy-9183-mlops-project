# feature_service/config.py (Updated OpenStackSettings for App Creds)

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, HttpUrl, ValidationError, model_validator
from typing import Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Note: OpenStack SDK's openstack.connect() reads standard OS_* variables directly.
# This Pydantic model is *only* for validating configuration within your Python application code.

class OpenStackSettings(BaseSettings):
    # This model will read OS_* environment variables
    model_config = SettingsConfigDict(env_prefix='OS_')

    # Common fields needed by SDK regardless of auth type
    auth_url: HttpUrl
    region_name: Optional[str] = None
    # project_id: Optional[str] = None # Project ID is needed for app creds

    # Fields for username/password auth (Optional in this model)
    username: Optional[str] = None
    password: Optional[str] = None
    # project_name: Optional[str] = None # Optional for SDK if project_id is used
    # project_domain_name: str = "Default" # Often has a default

    # Fields for application credential auth (Optional in this model)
    auth_type: str = Field("password", description="OpenStack authentication type (e.g., 'password', 'v3applicationcredential').")
    application_credential_id: Optional[str] = None
    application_credential_secret: Optional[str] = None
    user_domain_name: str = "Default" # Often has a default


    # Custom validation to ensure required fields are present based on auth_type
    @model_validator(mode='after')
    def check_auth_fields(self) -> 'OpenStackSettings':
        if self.auth_type == 'v3applicationcredential':
            if not self.application_credential_id:
                raise ValueError('OS_APPLICATION_CREDENTIAL_ID must be set for auth_type "v3applicationcredential"')
            if not self.application_credential_secret:
                 raise ValueError('OS_APPLICATION_CREDENTIAL_SECRET must be set for auth_type "v3applicationcredential"')
            # if not self.project_id:
            #      # Application credentials are project-scoped, project_id is required
            #      raise ValueError('OS_PROJECT_ID must be set for auth_type "application_credential"')

        elif self.auth_type in ['password', 'v3oidcpassword']: # Add other types as needed
            if not self.username:
                raise ValueError(f'OS_USERNAME must be set for auth_type "{self.auth_type}"')
            if not self.password:
                raise ValueError(f'OS_PASSWORD must be set for auth_type "{self.auth_type}"')
            if not self.project_name and not self.project_id:
                 # Need either project_name or project_id for project-scoped auth
                 raise ValueError(f'Either OS_PROJECT_NAME or OS_PROJECT_ID must be set for auth_type "{self.auth_type}"')

        # Add checks for other auth types if necessary
        elif self.auth_type not in ['token', 'v2password', 'v3password', 'v3token', 'v3oidcpassword', 'v3applicationcredential']:
             # Allow openstack.connect() to handle unknown types, maybe just log a warning
            logger.warning(f"Unknown OS_AUTH_TYPE '{self.auth_type}'. Validation skipped for auth-specific fields.")

        return self

class FeatureServiceSettings(BaseSettings):
    # This main settings model reads the FS_OPENSTACK_SWIFT_CONTAINER_NAME directly
    # and contains the nested OpenStackSettings which reads OS_* variables.
    # model_config = SettingsConfigDict(env_prefix='FS_') # REMOVE or adjust prefix if needed

    # Field specific to Feature Service, read with FS_OPENSTACK_ prefix
    model_config = SettingsConfigDict(env_prefix='FS_OPENSTACK_') # Add prefix here

    swift_container_name: str = Field(..., description="Name of the OpenStack Swift container.") # Reads FS_OPENSTACK_SWIFT_CONTAINER_NAME

    # Nested OpenStack authentication settings
    # This model will look for OS_* variables regardless of the parent prefix
    openstack: OpenStackSettings = Field(default_factory=OpenStackSettings) # Instantiate the nested settings


# Instantiate the settings when the module is imported
settings = FeatureServiceSettings()