from pydantic import BaseModel, Field

from agrivibe.agents.fieldoracle.schemas import CitedSentence
from agrivibe.models.knowledge import Source


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    question: str
    found: bool
    answer: str
    sentences: list[CitedSentence]
    sources: list[Source]
    searches: list[str]