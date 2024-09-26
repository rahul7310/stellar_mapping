from typing import Optional
from pydantic import BaseModel
from fastapi import UploadFile

class QueryRequest(BaseModel):
    query: str
    image: Optional[UploadFile] = None
