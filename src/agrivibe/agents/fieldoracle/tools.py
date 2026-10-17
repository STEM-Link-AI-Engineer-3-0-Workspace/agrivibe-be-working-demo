import logging

from langchain.tools import tool
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

from agrivibe.db.vectorstore import search_chunks, Category


def format_passages(docs: list[Document]) -> str:
    """Formats retrieved chunks as text for the model to read.

    Args:
        docs: The retrieved chunks.

    Returns:
        One block per chunk, headed by `[id] source, page N` so the model
        has an exact id to cite.
    """
    if not docs:
        return "No passages found."
    blocks = []
    for doc in docs:
        meta = doc.metadata
        header = f"[{doc.id}] {meta['source']}, page {meta['page']}"
        blocks.append(header + "\n" + doc.page_content)
    return "\n\n".join(blocks)


def make_search_tool(vector_store: VectorStore, k: int):

    @tool(response_format="content_and_artifact")
    def search_knowledge(query: str, category: Category | None = None):
        """Search the FieldOracle knowledge base for passages about a question.

        Args:
            query: What to search for, written as a short search query, not the farmer's full message.
            category: Limit the search to one document class. cultivation_guide covers how to grow a
                crop. policy covers government circulars, subsidies and mandates. farm_record covers
                what happened on a specific registered field. Leave empty to search everything.

        Returns:
            The passages as text for the model.
        """
        docs = [
            doc
            for doc, score in search_chunks(
                vector_store=vector_store, query=query, k=k, category=category
            )
        ]

        return format_passages(docs), docs

    return search_knowledge


# docs = [(Document(id="1, page_content="This is a test document.", metadata={"source": "test", "page": 1}), 0.95), (Document(id="2", page_content="This is another test document.", metadata={"source": "test", "page": 2}), 0.85), (Document(id="3", page_content="This is a third test document.", metadata={"source": "test", "page": 3}), 0.75) ...]

# [1] test, page 1
# This is a test document.

# [2] test, page 2
# This is another test document.

# [3] test, page 3
# This is a third test document.
