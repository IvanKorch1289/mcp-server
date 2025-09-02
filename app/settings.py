import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    giga_api_key: str = os.getenv("GIGA_API_KEY")

    # Tarantool
    tarantool_host: str = os.getenv("TARANTOOL_HOST", "localhost")
    tarantool_port: int = int(os.getenv("TARANTOOL_PORT", 3302))
    tarantool_user: str = os.getenv("TARANTOOL_USER", "admin")
    tarantool_password: str = os.getenv("TARANTOOL_PASSWORD", "password")

    # InfoSphere
    infosphere_login: str = os.getenv("INFOSPHERE_LOGIN")
    infosphere_password: str = os.getenv("INFOSPHERE_PASSWORD")
    infosphere_url: str = "https://i-sphere.ru/2.00/"

    # DaData
    dadata_api_key: str = os.getenv("DADATA_API_KEY")
    dadata_url: str = (
        "https://suggestions.dadata.ru/suggestions/api/4_1/rs/findById/party"
    )

    # Casebook
    casebook_api_key: str = os.getenv("CASEBOOK_API_KEY")

    class Config:
        env_file = ".env"


settings = Settings()
