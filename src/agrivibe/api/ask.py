from typing import Annotated

from fastapi import APIRouter, Depends

from agrivibe.models.ask import AskRequest, AskResponse
from agrivibe.service.answering import AnswerService
from agrivibe.dependencies import get_answer_service


router = APIRouter(tags=["agents"])

@router.post("/ask")
def ask(request: AskRequest, service: Annotated[AnswerService, Depends(get_answer_service)]) -> AskResponse:
    return service.ask(request.question)