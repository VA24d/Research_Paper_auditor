from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, read from AUDITOR_* environment variables and `.env`.

    Relative paths resolve against the working directory, so commands are
    expected to run from the repository root.
    """

    model_config = SettingsConfigDict(
        env_prefix="AUDITOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    grobid_url: str = "http://localhost:8070"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
