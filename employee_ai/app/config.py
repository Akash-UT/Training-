from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # Gemini
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "employee_ai"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # Qdrant Cloud
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "company_policies"

    # Data
    excel_path: Path = BASE_DIR / "data" / "employee_leave.xlsx"
    document_path: Path = BASE_DIR / "data" / "document.pdf"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:"
            f"{self.postgres_password}@"
            f"{self.postgres_host}:"
            f"{self.postgres_port}/"
            f"{self.postgres_db}"
        )


settings = Settings()