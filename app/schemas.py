from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# --- Common Schemas ---
class WebSource(BaseModel):
    title: str
    link: Optional[str] = None

class RetrievedSnippetPreview(BaseModel):
    metadata: Dict[str, Any]
    content_preview: str
    retrieval_score: Optional[float] = None # Score from initial vector search (e.g., FAISS L2 distance)
    llm_relevance_score: Optional[float] = None # Score from LLM evaluation

# --- Health Check ---
class HealthCheckResponse(BaseModel):
    status: str = "OK"
    app_name: str
    ollama_refiner_model: str
    ollama_helper_model: str
    ollama_base_url: str
    vector_store_status: Dict[str, Any]

# --- Code Input ---
class CodeInput(BaseModel):
    code_content: str = Field(..., description="The actual code content to store.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata (e.g., language, source_file, description).")

class CodeInputResponse(BaseModel):
    message: str
    source_id: Optional[str] = None
    chunk_ids_added: List[str]

# --- RAG Query Schemas ---
class TaskQuery(BaseModel):
    task_description: str = Field(..., description="Natural language description of the coding task.")
    language_hint: Optional[str] = Field("python", description="Optional hint for the desired programming language.")
    top_k_retrieval: int = Field(3, ge=1, le=10, description="Number of code snippets to retrieve.")
    # Let's keep it optional for potential pre-filtering in vector store if needed.
    # pre_filter_score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional FAISS relevance score threshold for initial retrieval.")

class SimilarCodeQuery(BaseModel):
    code_snippet: str = Field(..., description="The code snippet to find similar examples for.")
    refinement_task_description: Optional[str] = Field(None, description="Optional task description to refine the found similar code.")
    language_hint: Optional[str] = Field("python", description="Optional hint for the desired programming language for refinement.")
    top_k_retrieval: int = Field(3, ge=1, le=10)
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="FAISS relevance score threshold for similarity search.")

# --- RAG Response Schemas ---
class RefinedCodeResponse(BaseModel):
    query_task: str
    crag_action_taken: str = Field(..., description="Indicates the action CRAG took (e.g., 'retrieval_high_confidence', 'web_search_low_confidence', 'combined_ambiguous', 'web_search_only')")
    refined_code: str
    retrieved_context_count: int # Number of snippets initially retrieved
    retrieved_snippets_preview: List[RetrievedSnippetPreview] # <<< Updated schema type
    web_sources: List[WebSource] = Field([], description="List of web sources used if web search was performed.")

class RetrievedCodeSnippet(BaseModel): # <<< Renamed from SimilarCodeResponse's internal one
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None # FAISS score

class SimilarCodeResponse(BaseModel):
    query_code_preview: str
    similar_snippets: List[RetrievedCodeSnippet]
    # If refinement occurred based on similar code context:
    refined_code: Optional[str] = None
    refinement_task: Optional[str] = None
    web_sources: List[WebSource] = Field([], description="List of web sources used if refinement required web search.") # <<< Added
    crag_action_taken: Optional[str] = Field(None, description="Action taken during optional refinement step.") # <<< Added