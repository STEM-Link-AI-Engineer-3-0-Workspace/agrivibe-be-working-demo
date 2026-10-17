from functools import lru_cache

from langchain_pinecone import PineconeVectorStore

from agrivibe.agents.fieldoracle.agent import build_fieldoracle_agent
from agrivibe.service.knowledge import KnowledgeService
from agrivibe.service.answering import AnswerService
from agrivibe.config import Settings
from agrivibe.db.vectorstore import create_vector_store
from agrivibe.core.llm import create_chat_model

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

@lru_cache
def get_answer_service() -> AnswerService:
    settings = get_settings()
    rag_agent = build_fieldoracle_agent(
        create_chat_model(settings), get_vector_store(), k=settings.search_k
    )
    return AnswerService(rag_agent)