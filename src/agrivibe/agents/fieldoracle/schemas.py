from pydantic import BaseModel, Field


class CitedSentence(BaseModel):
    """One sentence of the answer and every passage it draws on."""

    sentence: str = Field(description="One sentence of the answer. It may combine facts from several passages.")
    citations: list[str] = Field(
        description="The ids of every passage this sentence uses, copied exactly from the search results. At least one."
    )


class CitedAnswer(BaseModel):
    """The final answer: sentences written only from retrieved passages, each with its citations."""

    found: bool = Field(
        description="True only if the retrieved passages answer the question. False if the knowledge base does not cover it."
    )
    sentences: list[CitedSentence] = Field(
        description="The answer, one sentence per item, each citing one or more passages. Empty when found is false."
    )
