from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    mlflow_tracking_uri: Optional[str] = None

    class Config:
        env_prefix = ""
        env_file = ".env"
        extra = "ignore"


settings = Settings()

