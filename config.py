from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env with override=True so project-specific credentials
# take precedence over any global system environment variables
load_dotenv(dotenv_path=".env", override=True)


class Settings(BaseSettings):
    # Denodo REST API (primary data source)
    denodo_base_url_allparts: str = "https://acldtpltfrm-dev:9443/server/dx_ce/ws_allparts_info_for_ce_app"
    denodo_base_url_manufacture: str = "https://acldtpltfrm-dev:9443/server/dx_ce/iv_plm_zagile_manufacture_ce_app"
    denodo_username: str = ""
    denodo_password: str = ""
    denodo_timeout: int = 60
    denodo_page_size: int = 5000
    denodo_enabled: bool = True

    # Power BI / Analysis Services (fallback)
    pbi_connection_string: str = "Data Source=localhost:52635;Application Name=MCP-PBIModeling"
    pbi_connection_name: str = "PBIDesktop-2026_plm_alparts-52635"

    # Azure OpenAI
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_deployment: str = "gpt-5.4"
    azure_openai_api_version: str = "2024-12-01-preview"

    # Batch settings
    batch_size_dev: int = 100
    batch_size_prod: int = 1000
    env: str = "dev"  # "dev" | "prod"

    # Azure OpenAI Embedding
    azure_openai_embedding_deployment: str = "text-embedding-3-small"
    azure_openai_embedding_api_version: str = "2024-02-01"

    # Fuzzy matching
    top_k_similar: int = 5
    min_similarity_score: float = 0.3

    # Vector DB fallback
    vector_db_top_k: int = 10

    # KPI
    kpi_data_dir: str = "data/kpi"

    # Paths
    output_dir: str = "data/results"
    batch_dir: str = "data/batches"

    # Target cache
    target_cache_dir: str = "data/cache"

    @property
    def batch_size(self) -> int:
        return self.batch_size_dev if self.env == "dev" else self.batch_size_prod

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
