from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Industrial Pricing Engine"
    app_env: str = "local"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    data_dir: Path = Path("./out")
    artifacts_dir: Path = Path("./artifacts")
    model_path: Path = Path("./artifacts/pricing_model.joblib")
    anomaly_path: Path = Path("./artifacts/anomaly_model.joblib")
    metadata_path: Path = Path("./artifacts/pricing_metadata.json")

    jwt_secret_key: str = "change-me-super-secret"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    mssql_host: str = "sqlserver"
    mssql_port: int = 1433
    mssql_db: str = "irge"
    mssql_user: str = "sa"
    mssql_password: str = "YourStrong!Passw0rd"
    mssql_driver: str = "ODBC Driver 18 for SQL Server"

    hf_token: str | None = None
    hf_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_base_url: str | None = None
    hf_timeout_seconds: int = 60

    enable_hf_agents: bool = True
    enable_auth: bool = False
    default_role: str = "analyst"

    prometheus_enabled: bool = True
    llm_front_password: str | None = None


settings = Settings()
