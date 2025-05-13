import logging
from typing import List, Dict, Any, Optional, Tuple

from app.core.vector_store import CodeVectorStore
from app.core.llm_services import CRAGLLMServices # <<< Changed import
from app import crud
from app.schemas import RefinedCodeResponse, SimilarCodeResponse, RetrievedCodeSnippet, WebSource # <<< Added WebSource
from app.core.config import settings # <<< Import settings for thresholds

logger = logging.getLogger(__name__)

class CodeRAGService:
    def __init__(self, vector_store: CodeVectorStore, llm_services: CRAGLLMServices): # <<< Changed injector
        self.vector_store = vector_store
        self.llm_services = llm_services # <<< Use the combined LLM services

    async def process_task_query(
        self,
        task_description: str,
        language_hint: Optional[str],
        top_k_retrieval: int,
        score_threshold: Optional[float]=0.2, # Note: This FAISS threshold is different from CRAG relevance threshold
    ) -> RefinedCodeResponse:
        logger.info(f"Processing CRAG task query: '{task_description[:100]}...'")

        # 1. Initial Retrieval from Vector Store
        retrieved_snippets_raw = crud.retrieve_code_from_vector_store(
            vector_store=self.vector_store,
            query=task_description,
            top_k=top_k_retrieval
        )

        if not retrieved_snippets_raw:
            logger.warning("Initial retrieval found no documents. Proceeding to web search only.")
            # Directly go to web search as if relevance was < low threshold
            final_context_str = "No relevant documents found in local database."
            web_knowledge_points, web_sources_tuples = self.llm_services.perform_web_search(task_description)
            if web_knowledge_points:
                final_context_str = "Context from Web Search:\n" + "\n".join([f"- {p}" for p in web_knowledge_points])
            web_sources_schema = [WebSource(title=title, link=link) for title, link in web_sources_tuples]
            action_taken = "web_search_only"

        else:
            # 2. Evaluate Relevance of Retrieved Docs using LLM
            logger.debug(f"Evaluating relevance of {len(retrieved_snippets_raw)} retrieved snippets...")
            eval_scores = []
            evaluated_snippets = [] # Store snippets with their scores
            for snippet in retrieved_snippets_raw:
                # Only evaluate if content exists
                if snippet.get('content'):
                    score = self.llm_services.evaluate_relevance(
                        query=task_description,
                        document=snippet['content']
                    )
                    eval_scores.append(score)
                    evaluated_snippets.append({**snippet, 'llm_relevance_score': score})
                else:
                    eval_scores.append(0.0) # Treat snippets without content as irrelevant
                    evaluated_snippets.append({**snippet, 'llm_relevance_score': 0.0})

            if not eval_scores: # Should not happen if retrieved_snippets_raw was not empty, but safety check
                 max_score = 0.0
            else:
                 max_score = max(eval_scores)
            logger.info(f"Relevance evaluation completed. Max score: {max_score:.2f}")

            # 3. CRAG Decision Logic
            best_retrieved_doc_content: Optional[str] = None
            best_retrieved_doc_metadata: Dict[str, Any] = {}
            if eval_scores:
                best_doc_index = eval_scores.index(max_score)
                best_retrieved_doc_content = evaluated_snippets[best_doc_index].get('content')
                best_retrieved_doc_metadata = evaluated_snippets[best_doc_index].get('metadata', {})

            web_sources_schema = [] # Initialize

            if max_score >= settings.CRAG_RELEVANCE_THRESHOLD_HIGH:
                logger.info(f"Action: Using highly relevant retrieved document (Score: {max_score:.2f}).")
                final_context_str = f"Context from Database (highly relevant):\n```\n{best_retrieved_doc_content}\n```"
                action_taken = "retrieval_high_confidence"

            elif max_score < settings.CRAG_RELEVANCE_THRESHOLD_LOW:
                logger.info(f"Action: Retrieved documents have low relevance (Max Score: {max_score:.2f}). Performing web search.")
                web_knowledge_points, web_sources_tuples = self.llm_services.perform_web_search(task_description)
                if not web_knowledge_points and best_retrieved_doc_content:
                    # If web search fails, but we have a (low relevance) doc, maybe use it? Or provide empty context?
                    # Let's provide empty context based on the original CRAG idea for "Incorrect"
                    final_context_str = "Context from Web Search (No results found or error)."
                    logger.warning("Web search failed or returned no results for low relevance case.")
                elif not web_knowledge_points: # Web search failed, no retrieved doc either or it was empty
                    final_context_str = "Context from Web Search (No results found or error)."
                else:
                     final_context_str = "Context from Web Search:\n" + "\n".join([f"- {p}" for p in web_knowledge_points])
                web_sources_schema = [WebSource(title=title, link=link) for title, link in web_sources_tuples]
                action_taken = "web_search_low_confidence"

            else: # Ambiguous case: Low < Score < High
                logger.info(f"Action: Ambiguous relevance (Score: {max_score:.2f}). Combining retrieved and web search.")
                # Refine/Extract from the best retrieved document
                retrieved_knowledge = []
                if best_retrieved_doc_content: # Ensure content exists
                     retrieved_knowledge = self.llm_services.extract_key_info(
                         query=task_description, document=best_retrieved_doc_content
                         )

                # Perform web search
                web_knowledge_points, web_sources_tuples = self.llm_services.perform_web_search(task_description)
                web_sources_schema = [WebSource(title=title, link=link) for title, link in web_sources_tuples]

                # Combine knowledge
                combined_knowledge = []
                if retrieved_knowledge:
                    combined_knowledge.append("--- Key Points from Retrieved Document ---")
                    combined_knowledge.extend([f"- {p}" for p in retrieved_knowledge])
                if web_knowledge_points:
                    if combined_knowledge: combined_knowledge.append("\n--- Key Points from Web Search ---")
                    else: combined_knowledge.append("--- Key Points from Web Search ---")
                    combined_knowledge.extend([f"- {p}" for p in web_knowledge_points])

                if not combined_knowledge:
                     final_context_str = "No usable context found from retrieval or web search in ambiguous case."
                     logger.warning("Both retrieval extraction and web search yielded no key points for ambiguous case.")
                else:
                     final_context_str = "\n".join(combined_knowledge)
                action_taken = "combined_ambiguous"

        # 4. Final Code Refinement using the determined context
        refined_code_str = self.llm_services.refine_code(
            task_description=task_description,
            context_str=final_context_str, # Pass the context decided by CRAG
            language_hint=language_hint
        )

        # Prepare preview of initially retrieved snippets (before CRAG logic kicked in fully)
        snippets_preview = [
            {
                "metadata": s.get("metadata", {}),
                "content_preview": s.get("content", "")[:50] + "...",
                "retrieval_score": s.get("score"), # Original FAISS score if available
                "llm_relevance_score": s.get("llm_relevance_score") # Score from LLM evaluator
            } for s in evaluated_snippets # Use the evaluated snippets list
        ] if 'evaluated_snippets' in locals() else [] # Handle case where retrieval failed entirely

        return RefinedCodeResponse(
            query_task=task_description,
            crag_action_taken=action_taken, # <<< Added field
            refined_code=refined_code_str,
            retrieved_context_count=len(retrieved_snippets_raw), # Count before evaluation/filtering
            retrieved_snippets_preview=snippets_preview,
            web_sources=web_sources_schema # <<< Added field
        )

    async def process_similar_code_query( # <<< Updated this method too for consistency
        self,
        code_snippet: str,
        refinement_task_description: Optional[str],
        language_hint: Optional[str],
        top_k_retrieval: int,
        score_threshold: Optional[float],
    ) -> SimilarCodeResponse:
        logger.info(f"Processing similar code query for snippet (preview): '{code_snippet[:100]}...'")

        # Retrieval based on the code snippet itself
        similar_docs_raw = crud.retrieve_code_from_vector_store(
            vector_store=self.vector_store,
            query=code_snippet,
            top_k=top_k_retrieval,
            score_threshold=score_threshold # Use score_threshold here as pre-filter maybe useful
        )

        similar_snippets_typed: List[RetrievedCodeSnippet] = [
             RetrievedCodeSnippet(content=doc.get("content",""), metadata=doc.get("metadata",{}), score=doc.get("score"))
             for doc in similar_docs_raw
        ]

        refined_code_str: Optional[str] = None
        web_sources_schema: List[WebSource] = []
        final_context_str = "No context provided."
        crag_action = "N/A (Refinement not requested)"

        if refinement_task_description:
            # If refinement is requested, we *could* apply CRAG logic here too,
            # evaluating the relevance of the *found similar code* to the *refinement task*.
            # For simplicity now, let's just use the found similar code as context directly.
            # A more advanced version could run the full CRAG pipeline based on the refinement task.

            logger.info(f"Refining similar code results based on task: '{refinement_task_description[:100]}...'")
            crag_action = "refinement_using_similar_code_context" # Simplified action

            if similar_docs_raw:
                # Format the similar code snippets as context
                context_parts = []
                for i, snippet in enumerate(similar_docs_raw):
                    meta_info = f"Source: {snippet['metadata'].get('source_id', 'unknown')}"
                    context_parts.append(f"-- Similar Snippet {i+1} --\n{meta_info}\nCode:\n{snippet['content']}\n-- End Snippet {i+1} --")
                final_context_str = "\n\n".join(context_parts)
            else:
                logger.warning("No similar code found to use as context for refinement task. Will attempt web search or from scratch.")
                # Optionally perform only web search here if no similar code was found
                web_knowledge_points, web_sources_tuples = self.llm_services.perform_web_search(refinement_task_description)
                if web_knowledge_points:
                    final_context_str = "Context from Web Search (No similar code found):\n" + "\n".join([f"- {p}" for p in web_knowledge_points])
                    web_sources_schema = [WebSource(title=title, link=link) for title, link in web_sources_tuples]
                    crag_action = "refinement_web_search_only (no similar found)"
                else:
                    final_context_str = "No similar code snippets found and web search failed/returned nothing."
                    crag_action = "refinement_from_scratch (no context)"

            # Call the refiner with the determined context
            refined_code_str = self.llm_services.refine_code(
                task_description=refinement_task_description,
                context_str=final_context_str,
                language_hint=language_hint
            )

        return SimilarCodeResponse(
            query_code_preview=code_snippet[:100] + "...",
            similar_snippets=similar_snippets_typed,
            refined_code=refined_code_str,
            refinement_task=refinement_task_description,
            web_sources=web_sources_schema, # <<< Added
            crag_action_taken=crag_action, # <<< Added
        )