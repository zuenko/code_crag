from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class HealthCheckResponse(BaseModel):
    status: str = "OK"
    app_name: str
    ollama_status: str
    vector_store_status: Dict[str, Any]

class CodeInput(BaseModel):
    code_content: str = Field(..., description="The actual code content to store.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata (e.g., language, source_file, description).")

class CodeInputResponse(BaseModel):
    message: str
    source_id: Optional[str] = None # ID of the original code snippet if metadata contained it or one was generated
    chunk_ids_added: List[str]

class TaskQuery(BaseModel):
    task_description: str = Field(..., description="Natural language description of the coding task.")
    language_hint: Optional[str] = Field("python", description="Optional hint for the desired programming language.")
    top_k_retrieval: int = Field(3, ge=1, le=10, description="Number of code snippets to retrieve for context.")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum relevance score for retrieved snippets.")

class SimilarCodeQuery(BaseModel):
    code_snippet: str = Field(..., description="The code snippet to find similar examples for.")
    # Optional: If refinement is also needed based on a task for the similar code
    refinement_task_description: Optional[str] = Field(None, description="Optional task description to refine the found similar code.")
    language_hint: Optional[str] = Field("python", description="Optional hint for the desired programming language for refinement.")
    top_k_retrieval: int = Field(3, ge=1, le=10)
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)

class RetrievedCodeSnippet(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None

class RefinedCodeResponse(BaseModel):
    query_task: str
    refined_code: str
    retrieved_context_count: int
    retrieved_snippets_preview: List[Dict[str, Any]] # e.g., metadata and start of content

class SimilarCodeResponse(BaseModel):
    query_code_preview: str
    similar_snippets: List[RetrievedCodeSnippet]
    refined_code: Optional[str] = None # If refinement_task_description was provided
    refinement_task: Optional[str] = None
