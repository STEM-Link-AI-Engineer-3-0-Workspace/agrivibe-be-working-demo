from typing import Annotated

from fastapi import APIRouter, UploadFile, Form, HTTPException, Depends

from agrivibe.db.vectorstore import Category
from agrivibe.service.knowledge import KnowledgeService
from agrivibe.dependencies import get_knowledge_service
from agrivibe.models.knowledge import SearchRequest, SearchResponse

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

Service = Annotated[KnowledgeService, Depends(get_knowledge_service)]

@router.post("/index", status_code=201)
def index_document(
    file: UploadFile,
    category: Annotated[Category, Form()],
    service: Service,
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files can be indexed.")

    try:
        service.index_pdf(file.filename, file.file.read(), category)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

@router.post("/search")
def search(request: SearchRequest, service: Service) -> SearchResponse:
    """Plain vector search, no model involved. Useful for checking what retrieval returns."""
    return service.search(request.query, request.k, request.category)
