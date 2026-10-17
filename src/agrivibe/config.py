from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: SecretStr
    pinecone_api_key: SecretStr

    chat_model: str = "openai:gpt-5.6-terra"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    pinecone_index: str = "fieldoracle"
    pinecone_cloud: str ="aws"
    pinecone_region: str = "us-east-1"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    search_k: int = 20