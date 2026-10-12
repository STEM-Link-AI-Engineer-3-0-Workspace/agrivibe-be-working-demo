import re
import hashlib
from io import BytesIO
from pathlib import Path

from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

from agrivibe.db.vectorstore import Category, search_chunks
from agrivibe.models.knowledge import IndexResponse, SearchResponse, SearchHit


def make_doc_id(filename: str, data: bytes) -> str:
    """Builds a stable id for a document from its name and contents.

    The same file always gets the same id, so re-uploading a PDF overwrites its
    chunks instead of duplicating them.

    Args:
        filename: The uploaded file's name.
        data: The file's bytes.

    Returns:
        The slugified file stem plus the first six hex digits of its SHA-256,
        for example `paddy-cultivation-guide-a29d87`.
    """
    stem = re.sub(r"[^a-z0-9]+", "-", Path(filename).stem.lower()).strip("-")
    digest = hashlib.sha256(data).hexdigest()[:6]
    return f"{stem}-{digest}"


def read_pdf(data: bytes, metadata: dict) -> list[Document]:
    pdf = PdfReader(BytesIO(data))

    return [
        Document(
            page_content=page.extract_text(), metadata={**metadata, "page": number}
        )
        for number, page in enumerate(pdf.pages, start=1)
        if (page.extract_text() or "").strip()
    ]


class KnowledgeService:

    def __init__(self, vector_store: VectorStore, chunk_size: int, chunk_overlap: int):
        self.vector_store = vector_store
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def index_pdf(self, filename: str, data: bytes, category: Category):
        doc_id = make_doc_id(filename, data)

        pages = read_pdf(
            data, {"doc_id": doc_id, "source": filename, "category": category}
        )

        if not pages:
            raise ValueError("No text could be extracted. Is this a scanned PDF?")

        chunks = self.splitter.split_documents(pages)

        for number, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = number
            chunk.id = f"{doc_id}-{number}"

        self.vector_store.add_documents(chunks)

        return IndexResponse(
            doc_id=doc_id,
            filename=filename,
            category=category,
            pages=len(pages),
            chunks=len(chunks),
        )

    def search(self, query: str, k: int, category: Category | None = None) -> SearchResponse:
        hits = [
            SearchHit(
                id=doc.id,
                source=doc.metadata["source"],
                page=doc.metadata["page"],
                category=doc.metadata["category"],
                text=doc.page_content,
                score=score,
            )
            for doc, score in search_chunks(self.vector_store, query, k, category)
        ]
        return SearchResponse(hits=hits)
