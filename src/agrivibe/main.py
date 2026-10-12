from fastapi import FastAPI

from agrivibe.api import knowledge

app = FastAPI(
    title="AgriVibe API",
    description="AgriVibe is a multi-agent agriculture knowledge system for Sri Lanka",
    version="0.1.0",
)

app.include_router(knowledge.router)

@app.get("/health", tags=["System", "Health"])
def health() -> dict[str, str]:
    return {"status": "ok"}
