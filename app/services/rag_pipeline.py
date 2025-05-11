import logging
from typing import List, Dict, Any, Optional

from app.core.vector_store import CodeVectorStore
from app.core.llm_services import OllamaCodeRefiner
from app import crud
from app.schemas import RefinedCodeResponse, SimilarCodeResponse, RetrievedCodeSnippet

logger = logging.getLogger(__name__)


class CodeRAGService:
    def __init__(self, vector_store: CodeVectorStore, code_refiner: OllamaCodeRefiner):
        self.vector_store = vector_store
        self.code_refiner = code_refiner

    async def process_task_query(
            self,
            task_description: str,
            language_hint: Optional[str],
            top_k_retrieval: int,
            score_threshold: Optional[float],
    ) -> RefinedCodeResponse:
        logger.info(f"Processing task query: '{task_description[:100]}...'")

        retrieved_snippets = crud.retrieve_code_from_vector_store(
            vector_store=self.vector_store,
            query=task_description,
            top_k=top_k_retrieval,
            score_threshold=score_threshold
        )  # This returns List[Dict] with "content", "metadata", "score"

        refined_code_str = self.code_refiner.refine_code(
            task_description=task_description,
            retrieved_snippets=retrieved_snippets,
            language_hint=language_hint
        )

        # For preview, take metadata and first 50 chars of content
        snippets_preview = [
            {
                "metadata": s.get("metadata", {}),
                "content_preview": s.get("content", "")[:50] + "...",
                "score": s.get("score")
            } for s in retrieved_snippets
        ]

        return RefinedCodeResponse(
            query_task=task_description,
            refined_code=refined_code_str,
            retrieved_context_count=len(retrieved_snippets),
            retrieved_snippets_preview=snippets_preview
        )

    async def process_similar_code_query(
            self,
            code_snippet: str,
            refinement_task_description: Optional[str],
            language_hint: Optional[str],
            top_k_retrieval: int,
            score_threshold: Optional[float],
    ) -> SimilarCodeResponse:
        logger.info(f"Processing similar code query for snippet (preview): '{code_snippet[:100]}...'")

        similar_docs_raw = crud.retrieve_code_from_vector_store(
            vector_store=self.vector_store,
            query=code_snippet,  # Use the code itself as the query
            top_k=top_k_retrieval,
            score_threshold=score_threshold
        )

        similar_snippets_typed: List[RetrievedCodeSnippet] = [
            RetrievedCodeSnippet(**doc) for doc in similar_docs_raw
        ]

        refined_code_str: Optional[str] = None
        if refinement_task_description:
            logger.info(f"Refining similar code based on task: '{refinement_task_description[:100]}...'")
            # Pass the retrieved similar snippets as context for refinement
            refined_code_str = self.code_refiner.refine_code(
                task_description=refinement_task_description,
                retrieved_snippets=similar_docs_raw,  # pass the raw list of dicts
                language_hint=language_hint
            )

        return SimilarCodeResponse(
            query_code_preview=code_snippet[:100] + "...",
            similar_snippets=similar_snippets_typed,
            refined_code=refined_code_str,
            refinement_task=refinement_task_description
        )
