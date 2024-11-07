from fastapi import FastAPI
from schema.skeleton import QueryRequest
import uvicorn
from src.analytics.text_analytics.src.stellar_llm_generation.core.generation.llm_generation import LLMGenerator

app = FastAPI()

generator = LLMGenerator()

@app.post("/text/chat/generate")
async def rag(query_request: QueryRequest):
    # Retrieval
    pass
    # ReRanking
    pass
    # Generation
    model_response = generator.generate(query_request.messages)
    return {"model_response": model_response}



