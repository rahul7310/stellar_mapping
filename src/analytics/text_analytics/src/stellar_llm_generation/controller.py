from fastapi import FastAPI
from schema.skeleton import QueryRequest
import uvicorn

app = FastAPI()

@app.post("/rag")
async def rag(query_request: QueryRequest):
    return {"query": query_request.query, "image": query_request.image}

