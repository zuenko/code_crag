import logging
import traceback
from contextlib import asynccontextmanager
from typing import Annotated

from litestar import Litestar, get, post, MediaType
from litestar.datastructures import State
from litestar.exceptions import HTTPException
from litestar.status_codes import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_400_BAD_REQUEST

from app.core.logging_config import setup_logging
from app.core.vector_store import CodeVectorStore
from app.core.llm_services import CRAGLLMServices
from app.core.config import settings
from app.services.rag_pipeline import CodeRAGService
from app.schemas import (
    HealthCheckResponse, CodeInput, CodeInputResponse,
    TaskQuery, RefinedCodeResponse, SimilarCodeQuery, SimilarCodeResponse
)
from app import crud

# Call setup_logging at the module level to configure logging when the app starts.
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: Litestar):
    logger.info(f"Starting up {settings.APP_NAME}...")
    app.state.vector_store = None
    app.state.llm_services = None # Changed name
    app.state.rag_service = None

    try:
        vector_store_instance = CodeVectorStore(index_path=settings.FAISS_INDEX_PATH)
        app.state.vector_store = vector_store_instance
        logger.info(f"CodeVectorStore initialized. Path: {settings.FAISS_INDEX_PATH}. Stats: {vector_store_instance.get_stats()}")
    except Exception as e:
        logger.error(f"Failed to initialize CodeVectorStore: {e}", exc_info=True)

    try:
        # Initialize the combined LLM services class
        llm_services_instance = CRAGLLMServices(
             refiner_model_name=settings.OLLAMA_REFINER_MODEL,
             helper_model_name=settings.OLLAMA_HELPER_MODEL,
             base_url=settings.OLLAMA_BASE_URL
             )
        app.state.llm_services = llm_services_instance
        logger.info(f"CRAGLLMServices initialized. Refiner: {settings.OLLAMA_REFINER_MODEL}, Helper: {settings.OLLAMA_HELPER_MODEL}")
    except Exception as e:
        logger.error(f"Failed to initialize CRAGLLMServices: {e}", exc_info=True)

    # Initialize RAG Service if dependencies are met
    if app.state.vector_store and app.state.llm_services:
         app.state.rag_service = CodeRAGService(
             vector_store=app.state.vector_store,
             llm_services=app.state.llm_services # Pass combined services
         )
         logger.info("CodeRAGService initialized.")
    else:
         logger.error("CodeRAGService could not be initialized due to missing dependencies (vector_store or llm_services).")

    yield

    # Clean up resources if needed on shutdown
    if app.state.vector_store:
        app.state.vector_store.save_index()
    logger.info(f"Shutting down {settings.APP_NAME}...")

# --- API Endpoints ---

# Health check
@get("/health", media_type=MediaType.JSON, summary="Health Check", tags=["General"])
async def health_check(state: State) -> HealthCheckResponse:
    ollama_refiner_model = "N/A"
    ollama_helper_model = "N/A"
    ollama_base_url = "N/A"
    if state.llm_services and state.llm_services.refiner_llm:
        ollama_refiner_model = state.llm_services.refiner_model_name # Get from instance
        ollama_base_url = state.llm_services.base_url
    if state.llm_services and state.llm_services.helper_llm:
         ollama_helper_model = state.llm_services.helper_model_name
    vs_status = state.vector_store.get_stats() if state.vector_store else {"status": "Error: Vector Store not initialized"}

    return HealthCheckResponse(
        app_name=settings.APP_NAME,
        ollama_refiner_model=ollama_refiner_model, # <<< Updated
        ollama_helper_model=ollama_helper_model, # <<< Updated
        ollama_base_url=ollama_base_url, # <<< Updated
        vector_store_status=vs_status
    )

# Add code snippet
@post("/code", summary="Add Code Snippet", tags=["Code Storage"], status_code=201)
async def add_code_snippet(
    state: State, data: Annotated[CodeInput, ...]
) -> CodeInputResponse:
    if not state.vector_store:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="Vector store not initialized.")
    try:
        logger.debug(f"Received code snippet for addition. Metadata: {data.metadata}")
        chunk_ids = crud.add_code_to_vector_store(state.vector_store, data)
        source_id = data.metadata.get("source_id") if data.metadata else None
        if not chunk_ids and data.code_content.strip(): # Content was there but no chunks made
             logger.warning(f"Code was provided but no chunks were added. Source ID if any: {source_id}")
             # Potentially return an error or different message
        return CodeInputResponse(
            message=f"Code snippet processed. {len(chunk_ids)} chunk(s) handled.",
            source_id=source_id or (state.vector_store.vector_store.docstore._dict[state.vector_store.vector_store.index_to_docstore_id[0]].metadata.get("source_id") if chunk_ids and state.vector_store.vector_store else "N/A"), # Attempt to get a source_id if one was generated
            chunk_ids_added=chunk_ids
        )
    except Exception as e:
        logger.error(f"Error adding code snippet: {e}", exc_info=True)
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# Query by task description and get refined code
@post("/code/query/task", summary="Query by Task Description & Refine Code", tags=["RAG Query"])
async def query_by_task(
    state: State, data: Annotated[TaskQuery, ...]
) -> RefinedCodeResponse:
    if not state.rag_service:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="RAG service not initialized.")
    try:
        response = await state.rag_service.process_task_query(
            task_description=data.task_description,
            language_hint=data.language_hint,
            top_k_retrieval=data.top_k_retrieval
        )
        return response
    except Exception as e:
        logger.error(f"Error processing task query '{data.task_description[:50]}...': {e}", exc_info=True)
        logger.error(str(traceback.format_exc()))
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# Query by similar code and optionally refine
@post("/code/query/similar", summary="Find Similar Code & Optionally Refine", tags=["RAG Query"])
async def query_by_similar_code(
    state: State, data: Annotated[SimilarCodeQuery, ...]
) -> SimilarCodeResponse:
    if not state.rag_service:
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="RAG service not initialized.")
    if not data.code_snippet.strip():
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="Query code_snippet cannot be empty.")
    try:
        response = await state.rag_service.process_similar_code_query(
            code_snippet=data.code_snippet,
            refinement_task_description=data.refinement_task_description,
            language_hint=data.language_hint,
            top_k_retrieval=data.top_k_retrieval,
            score_threshold=data.score_threshold
        )
        return response
    except Exception as e:
        logger.error(f"Error processing similar code query: {e}", exc_info=True)
        raise HTTPException(status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# Create the Litestar application instance
app = Litestar(
    route_handlers=[health_check, add_code_snippet, query_by_task, query_by_similar_code],
    lifespan=[lifespan],
)

# For running with `python app/main.py` for local dev (though uvicorn is better)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=settings.LOG_LEVEL.lower())
