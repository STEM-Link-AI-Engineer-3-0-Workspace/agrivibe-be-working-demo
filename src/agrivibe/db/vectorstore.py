import os
from typing import Literal

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

from agrivibe.config import Settings

Category = Literal["cultivation_guide", "policy", "farm_record"]


def create_vector_store(settings: Settings) -> PineconeVectorStore:
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )

    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())

    if not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud, region=settings.pinecone_region
            ),
        )

    return PineconeVectorStore(
        index_name=settings.pinecone_index,
        embedding=embeddings,
        pinecone_api_key=settings.pinecone_api_key.get_secret_value(),
    )


def search_chunks(
    vector_store: VectorStore, query: str, k: int, category: Category | None = None
) -> list[tuple[Document, float]]:
    search_filter = {"category": category} if category else None
    return vector_store.similarity_search_with_score(
        query=query, k=k, filter=search_filter
    )
