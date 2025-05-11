import logging
from typing import List, Dict, Any, Optional

from app.core.vector_store import CodeVectorStore
from app.schemas import CodeInput

logger = logging.getLogger(__name__)

def add_code_to_vector_store(
    vector_store: CodeVectorStore, code_input: CodeInput
) -> List[str]:
    """Adds a code snippet to the vector store."""
    logger.info(f"Adding code to vector store. Metadata: {code_input.metadata}")
    # Ensure source_id is in metadata if it's important to track back
    metadata = code_input.metadata or {}
    # source_id = metadata.get("source_id") # Or generate one in vector_store
    chunk_ids = vector_store.add_code_snippet(code_input.code_content, metadata)
    return chunk_ids

def retrieve_code_from_vector_store(
    vector_store: CodeVectorStore,
    query: str,
    top_k: int,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Retrieves code snippets based on a text query."""
    logger.info(f"Retrieving top {top_k} code snippets for query: '{query[:100]}...'")
    results = vector_store.retrieve_similar_documents(query, k=top_k, score_threshold=score_threshold)
    return results
