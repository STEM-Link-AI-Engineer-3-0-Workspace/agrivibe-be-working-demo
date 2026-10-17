from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.documents import Document

from agrivibe.agents.fieldoracle.schemas import CitedAnswer, CitedSentence
from agrivibe.models.ask import AskResponse
from agrivibe.models.knowledge import Source


def to_source(doc: Document) -> Source:
    meta = doc.metadata
    return Source(id=doc.id, source=meta["source"], page=meta["page"], category=meta["category"], text=doc.page_content)

def retrieved_chunks(messages: list[BaseMessage]) -> dict[str, Document]:
    """Collects every chunk the search tool returned during one agent run.

    Args:
        messages: The agent's message history for the run.

    Returns:
        The retrieved chunks, keyed by chunk id.
    """
    chunks = {}
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == "search_knowledge":
            for doc in message.artifact or []:
                chunks[doc.id] = doc
    return chunks

def search_queries(messages: list[BaseMessage]) -> list[str]:
    """Lists the queries the agent chose to search for.

    Args:
        messages: The agent's message history for the run.

    Returns:
        The search queries, in the order the agent ran them.
    """
    return [
        call["args"]["query"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] == "search_knowledge"
    ]

"""
{
    "5": {
        "_id":
        "text":
        "metadata
    },
    "2": {}
}
"""

class AnswerService:
    """Answers questions with the RAG agent."""

    def __init__(self, rag_agent):
        """Initializes the service.

        Args:
            rag_agent: The agent built by `build_rag_agent`.
        """
        self.rag_agent = rag_agent

    def ask(self, question: str):

        result = self.rag_agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )

        answer: CitedAnswer = result["structured_response"]
        messages = result["messages"]

        found = answer.found
        sentences = answer.sentences
        cited = {chunk_id for s in sentences for chunk_id in s.citations}

        return AskResponse(
            question=question,
            found=found,
            answer=" ".join(f"{s.sentence} [{', '.join(s.citations)}]" for s in sentences),
            sentences=sentences,
            sources=[to_source(doc) for chunk_id, doc in retrieved_chunks(messages).items()],
            searches=search_queries(messages)
        )


# This is sentence 1 [4, 1, 3] This is Sentence 2 [2, 3, 5]