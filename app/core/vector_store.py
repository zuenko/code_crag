import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import uuid

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Or CodeSplitter
from langchain.docstore.document import Document as LangchainDocument

from app.core.config import settings

logger = logging.getLogger(__name__)


class CodeVectorStore:
    def __init__(
            self,
            index_path: Path = settings.FAISS_INDEX_PATH,
            embeddings_model_name: str = settings.EMBEDDINGS_MODEL_NAME,
            chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
    ):
        self.index_path: Path = index_path
        self.embeddings_model = OpenAIEmbeddings(
            model=embeddings_model_name, api_key=settings.OPENAI_API_KEY
        )
        # For code, consider RecursiveCharacterTextSplitter.from_language if language is known
        # or a more generic one if code can be in multiple languages.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,  # Can be True for more complex patterns
        )
        self.vector_store: Optional[FAISS] = None
        self._load_or_initialize()

    def _load_or_initialize(self):
        try:
            if self.index_path.exists() and any(self.index_path.iterdir()):
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                self.vector_store = FAISS.load_local(
                    folder_path=str(self.index_path),
                    embeddings=self.embeddings_model,
                    allow_dangerous_deserialization=True  # Required for FAISS
                )
                logger.info(
                    f"FAISS index loaded. Total documents: {self.vector_store.index.ntotal if self.vector_store.index else 0}")
            else:
                logger.info(f"FAISS index not found at {self.index_path}. Initializing new store.")
                self.index_path.mkdir(parents=True, exist_ok=True)
                # FAISS needs at least one document to initialize. So, we initialize it on first add.
                self.vector_store = None
        except Exception as e:
            logger.error(f"Error loading or initializing FAISS index: {e}", exc_info=True)
            # Depending on desired behavior, might raise or try to re-initialize
            self.vector_store = None

    def add_code_snippet(self, code_content: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        if not code_content or not code_content.strip():
            logger.warning("Attempted to add empty code snippet.")
            return []

        doc_metadata = metadata.copy() if metadata else {}
        # Ensure a unique ID for the original source if not provided
        doc_metadata.setdefault('source_id', str(uuid.uuid4()))
        # Store original code length or other relevant info
        doc_metadata.setdefault('original_length', len(code_content))

        # Split the code content into manageable chunks
        texts = self.text_splitter.split_text(code_content)
        if not texts:
            logger.warning(
                f"Code content from source_id {doc_metadata['source_id']} resulted in no chunks after splitting.")
            return []

        langchain_docs: List[LangchainDocument] = []
        chunk_ids_added: List[str] = []

        for i, text_chunk in enumerate(texts):
            chunk_meta = doc_metadata.copy()
            chunk_meta['chunk_index'] = i
            chunk_id = str(uuid.uuid4())  # Unique ID for each chunk
            chunk_meta['chunk_id'] = chunk_id
            langchain_docs.append(LangchainDocument(page_content=text_chunk, metadata=chunk_meta))
            chunk_ids_added.append(chunk_id)

        if not langchain_docs:
            return []

        try:
            if self.vector_store is None:
                logger.info(
                    f"Initializing FAISS index with first document(s) from source_id {doc_metadata['source_id']}.")
                self.vector_store = FAISS.from_documents(langchain_docs, self.embeddings_model)
            else:
                added_doc_ids = self.vector_store.add_documents(langchain_docs)
                logger.debug(f"Added {len(added_doc_ids)} chunks to FAISS from source_id {doc_metadata['source_id']}.")

            self.save_index()  # Persist after adding
            logger.info(
                f"Added {len(langchain_docs)} chunk(s) for source_id {doc_metadata['source_id']}. Total docs: {self.vector_store.index.ntotal if self.vector_store else 0}")
            return chunk_ids_added
        except Exception as e:
            logger.error(f"Error adding documents to FAISS for source_id {doc_metadata['source_id']}: {e}",
                         exc_info=True)
            return []

    def retrieve_similar_documents(
            self, query_text: str, k: int = 5, score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        if self.vector_store is None:
            logger.warning("Vector store is not initialized. Cannot retrieve.")
            return []
        if not query_text or not query_text.strip():
            logger.warning("Empty query text for retrieval.")
            return []

        logger.debug(f"Retrieving top {k} documents for query: '{query_text[:100]}...'")
        try:
            if score_threshold is not None:
                docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(query_text, k=k)
                results = [
                    {"document": doc, "score": score} for doc, score in docs_and_scores if score >= score_threshold
                ]
            else:
                docs = self.vector_store.similarity_search(query_text, k=k)
                results = [{"document": doc, "score": None} for doc in docs]

            logger.info(f"Retrieved {len(results)} documents for query.")
            return [
                {"content": res["document"].page_content, "metadata": res["document"].metadata, "score": res["score"]}
                for res in results]
        except Exception as e:
            logger.error(f"Error during similarity search: {e}", exc_info=True)
            return []

    def save_index(self):
        if self.vector_store:
            try:
                self.vector_store.save_local(folder_path=str(self.index_path))
                logger.info(f"FAISS index saved to {self.index_path}")
            except Exception as e:
                logger.error(f"Error saving FAISS index to {self.index_path}: {e}", exc_info=True)
        else:
            logger.warning("Attempted to save an uninitialized vector store.")

    def get_stats(self) -> Dict[str, Any]:
        if self.vector_store and self.vector_store.index:
            return {"total_documents_in_index": self.vector_store.index.ntotal}
        return {"total_documents_in_index": 0, "status": "Not initialized or empty"}
