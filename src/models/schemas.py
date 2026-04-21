from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str]


class ProjectSettings(BaseModel):
    embedding_model: str
    rag_strategy: str
    agent_type: str
    chunks_per_search: int
    final_context_size: int
    similarity_threshold: float
    number_of_queries: int
    reranking_enabled: bool
    reranking_model: str
    vector_weight: float
    keyword_weight: float


class ProcessingStatus(str, Enum):
    UPLOADING = "uploading"
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    PARTITIONING = "partitioning"
    CHUNKING = "chunking"
    SUMMARISING = "summarising"
    VECTORIZATION = "vectorization"
    COMPLETED = "completed"
    

class FileUploadRequest(BaseModel):
    filename: str
    file_size: int
    file_type: str


class UrlAddRequest(BaseModel):
    url: str


class ChatCreate(BaseModel):
    title: str
    project_id: str


class QueryVariations(BaseModel):
    queries: List[str]


class SendMessageRequest(BaseModel):
    content: str

class UrlRequest(BaseModel):
    url: str = Field(..., description="The URL to process")


class MessageCreate(BaseModel):
    content: str = Field(..., description="The content of the message")


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"