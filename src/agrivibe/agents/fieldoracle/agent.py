from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from agrivibe.agents.fieldoracle.prompts import SYSTEM_PROMPT
from agrivibe.agents.fieldoracle.tools import make_search_tool
from agrivibe.agents.fieldoracle.schemas import CitedAnswer


def build_fieldoracle_agent(
    model: BaseChatModel, vector_store: VectorStore, k: int = 20
):
    
    agent = create_agent(
        model=model,
        tools=[make_search_tool(vector_store=vector_store, k=k)],
        system_prompt=SYSTEM_PROMPT,
        response_format=CitedAnswer,
        name="FieldOracle",
    )

    return agent


if __name__ == "__main__":
    from agrivibe.config import Settings
    from agrivibe.db.vectorstore import create_vector_store
    from langchain_openai import ChatOpenAI

    settings = Settings()
    vector_store = create_vector_store(settings=settings)
    model = ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=settings.openai_api_key.get_secret_value(),
        use_responses_api=True,
    )

    agent = build_fieldoracle_agent(model=model, vector_store=vector_store, k=5)

    question = "What is the best time to plant rice in Sri Lanka?"
    answer = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(answer)
