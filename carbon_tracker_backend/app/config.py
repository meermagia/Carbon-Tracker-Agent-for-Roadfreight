from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Carbon Tracker Backend"
    environment: str = "development"
    debug: bool = True

    # Database (explicit env mapping)
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "carbon_tracker"
    db_user: str = "postgres"
    db_password: str = "user"

    _base_dir = Path(__file__).resolve().parents[1]
    model_config = SettingsConfigDict(
        env_file=str(_base_dir / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache()
def get_settings() -> Settings:
    return Settings()