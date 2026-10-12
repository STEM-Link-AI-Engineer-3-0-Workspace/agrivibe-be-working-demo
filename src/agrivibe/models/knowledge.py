from pydantic import BaseModel, Field

from agrivibe.db.vectorstore import Category

class IndexResponse(BaseModel):
    """The result of indexing one PDF."""

    doc_id: str
    filename: str
    category: Category
    pages: int
    chunks: int

class Source(BaseModel):
    """A retrieved chunk and where it came from."""

    id: str
    source: str
    page: int
    category: Category
    text: str

class SearchHit(Source):
    """A search result: a chunk plus its similarity score."""

    score: float

class SearchRequest(BaseModel):
    """A plain vector search, optionally limited to one category."""

    query: str = Field(min_length=1)
    category: Category | None = None
    k: int = Field(default=4, ge=1, le=20)

class SearchResponse(BaseModel):
    """The hits of a search, best match first."""

    hits: list[SearchHit]