from functools import lru_cache

from langchain_pinecone import PineconeVectorStore

from agrivibe.service.knowledge import KnowledgeService
from agrivibe.config import Settings
from agrivibe.db.vectorstore import create_vector_store

@lru_cache
def get_settings() -> Settings:
    return Settings()

@lru_cache
def get_vector_store() -> PineconeVectorStore:
    settings = get_settings()
    return create_vector_store(settings)

@lru_cache
def get_knowledge_service() -> KnowledgeService:
    settings = get_settings()
    return KnowledgeService(
        vector_store=get_vector_store(),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )