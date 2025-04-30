from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, HttpUrl # Import HttpUrl for auth_url
from pathlib import Path
from typing import Optional # For optional region name

class OpenStackSettings(BaseSettings):
    """Settings for OpenStack connectivity."""
    model_config = SettingsConfigDict(env_prefix='OS_') # Reads OS_AUTH_URL etc.

    auth_url: HttpUrl = Field(..., description="OpenStack Identity (Keystone) endpoint URL.")
    project_name: str = Field(..., description="OpenStack project name.")
    project_domain_name: str = Field("Default", description="OpenStack project domain name (default: Default).")
    username: str = Field(..., description="OpenStack username.")
    user_domain_name: str = Field("Default", description="OpenStack user domain name (default: Default).")
    password: str = Field(..., description="OpenStack password.")
    region_name: Optional[str] = Field(None, description="OpenStack region name (optional).")
    swift_container_name: str = Field(..., description="Name of the OpenStack Swift container storing the data lake.")

class FeatureServiceSettings(BaseSettings):
    """Overall settings for the Feature Service."""
    # We no longer need data_lake_root as a local path
    # model_config = SettingsConfigDict(env_prefix='FS_') # Prefix is now handled by nested settings

    openstack: OpenStackSettings = OpenStackSettings()


# Load settings instance - reads from environment variables
try:
    settings = FeatureServiceSettings()
    print("Feature Service Settings loaded successfully.")
    # print sensitive info carefully, maybe only during dev
    # print(f"Swift Container: {settings.openstack.swift_container_name}")

except Exception as e:
    # Use standard logging here as structlog might not be fully configured yet
    logging.error(f"Failed to load settings: {e}", exc_info=True)
    raise # Re-raise to prevent service startup if config fails