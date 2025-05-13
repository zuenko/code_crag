import logging
from typing import List, Optional, Tuple
import json

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field as PydanticField # Renamed Field to avoid conflict
from langchain_community.tools import DuckDuckGoSearchResults

from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Helper Pydantic Schemas for Structured Output ---

class RelevanceScoreOutput(BaseModel):
    score: float = PydanticField(..., description="Relevance score between 0.0 and 1.0.", ge=0.0, le=1.0)
    reasoning: Optional[str] = PydanticField(None, description="Brief reasoning for the score.")

class RewrittenQueryOutput(BaseModel):
    rewritten_query: str = PydanticField(..., description="The rewritten query suitable for web search.")

class KeyPointsOutput(BaseModel):
    key_points: List[str] = PydanticField(..., description="A list of key points extracted from the document.")

# --- Helper Function for Safe JSON Parsing ---
def parse_search_results(results_string: str) -> List[Tuple[str, str]]:
    """ Safe parsing for DuckDuckGo results (which returns a stringified list) """
    if not isinstance(results_string, str) or not results_string.strip():
        return []
    try:
        # DDG often returns a string representation of a list of dicts
        # Example: "[{'title': '...', 'link': '...', 'snippet': '...'}, ...]"
        # We need to handle potential variations
        results_list = eval(results_string) # Using eval is risky, but often needed for DDG output format
                                            # Alternatives: ast.literal_eval or regex parsing
        if isinstance(results_list, list):
             return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results_list if isinstance(result, dict)]
        else:
             logger.warning(f"Could not parse DDG results string into a list: {results_string[:100]}...")
             return []
    except Exception as e:
        logger.error(f"Error parsing DDG search results string: {e}. String: {results_string[:100]}...", exc_info=True)
        # Attempt simpler parsing if eval fails - maybe it's already JSON-like?
        try:
            results = json.loads(results_string)
            if isinstance(results, list):
                 return [(result.get('title', 'Untitled'), result.get('link', '')) for result in results if isinstance(result, dict)]
        except json.JSONDecodeError:
            pass # Ignore if JSON parsing also fails
        return []

# --- Main LLM Service Class ---

class CRAGLLMServices:
    """
    Combines Ollama LLM functionalities needed for CRAG:
    - Code Refinement
    - Relevance Evaluation
    - Web Query Rewriting
    - Key Information Extraction
    - Web Search Tool
    """
    def __init__(
        self,
        refiner_model_name: str = settings.OLLAMA_REFINER_MODEL,
        helper_model_name: str = settings.OLLAMA_HELPER_MODEL,
        base_url: str = settings.OLLAMA_BASE_URL,
        temperature_refine: float = 0.1,
        temperature_helper: float = 0.0,
    ):
        self.base_url = base_url
        self.refiner_model_name = refiner_model_name
        self.helper_model_name = helper_model_name

        # Initialize LLMs (can be the same model object if names match)
        try:
            self.refiner_llm = Ollama(model=refiner_model_name, base_url=base_url, temperature=temperature_refine)
            self.refiner_llm.invoke("Ping") # Test connection
            logger.info(f"Ollama REFINER model '{refiner_model_name}' connected.")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama REFINER model '{refiner_model_name}': {e}", exc_info=True)
            raise ConnectionError(f"Ollama connection failed for REFINER model {refiner_model_name}") from e

        if refiner_model_name == helper_model_name:
            self.helper_llm = self.refiner_llm # Use the same instance
            logger.info("Using the same Ollama instance for helper tasks.")
        else:
            try:
                self.helper_llm = Ollama(model=helper_model_name, base_url=base_url, temperature=temperature_helper)
                self.helper_llm.invoke("Ping") # Test connection
                logger.info(f"Ollama HELPER model '{helper_model_name}' connected.")
            except Exception as e:
                logger.error(f"Failed to connect to Ollama HELPER model '{helper_model_name}': {e}", exc_info=True)
                # Decide how to handle: maybe fallback to refiner_llm?
                logger.warning(f"Falling back to use REFINER model '{refiner_model_name}' for helper tasks due to HELPER connection error.")
                self.helper_llm = self.refiner_llm

        # --- Initialize Tools and Chains ---
        self.web_search_tool = DuckDuckGoSearchResults(output_format="list")

        # 1. Relevance Evaluator Chain (Uses Helper LLM)
        self.relevance_parser = JsonOutputParser(pydantic_object=RelevanceScoreOutput)
        self.relevance_prompt = PromptTemplate(
            template="Evaluate the relevance of the following code snippet to the given task description. Provide a score between 0.0 (not relevant) and 1.0 (highly relevant) and a brief reasoning.\n"
                     "Task Description: {query}\n"
                     "Code Snippet:\n```\n{document}\n```\n"
                     "{format_instructions}",
            input_variables=["query", "document"],
            partial_variables={"format_instructions": self.relevance_parser.get_format_instructions()},
        )
        self.relevance_chain = self.relevance_prompt | self.helper_llm | self.relevance_parser

        # 2. Web Query Rewriter Chain (Uses Helper LLM)
        self.rewrite_parser = JsonOutputParser(pydantic_object=RewrittenQueryOutput)
        self.rewrite_prompt = PromptTemplate(
             template="Rewrite the following user query, which describes a coding task, into a concise and effective query suitable for a web search engine (like Google or DuckDuckGo) to find relevant code examples or solutions.\n"
                      "If there are particular framework, include it in a query \n"
                      "Original Query: {query}\n"
                      "{format_instructions}",
             input_variables=["query"],
             partial_variables={"format_instructions": self.rewrite_parser.get_format_instructions()},
        )
        self.rewrite_chain = self.rewrite_prompt | self.helper_llm | self.rewrite_parser

        # 3. Key Info Extractor Chain (Uses Helper LLM)
        self.extract_parser = JsonOutputParser(pydantic_object=KeyPointsOutput)
        self.extract_prompt = PromptTemplate(
             template="Read the following text, which contains results from a web search about a programming topic. Extract the key pieces of information, code snippets, or main ideas relevant to solving the original query. Present them as a list of concise bullet points.\n"
                      "Original Query (for context): {query}\n"
                      "Web Search Results Text:\n```\n{document}\n```\n"
                      "{format_instructions}",
             input_variables=["query", "document"],
             partial_variables={"format_instructions": self.extract_parser.get_format_instructions()},
        )
        self.extract_chain = self.extract_prompt | self.helper_llm | self.extract_parser

        # 4. Code Refiner Chain (Uses Refiner LLM)
        self.refine_output_parser = StrOutputParser()
        # Using the same robust prompt as before for refinement
        self.refine_prompt = PromptTemplate(
            input_variables=["task_description", "retrieved_code_snippets", "language_hint"],
            template="You are an expert AI programmer. Your goal is to generate a high-quality, "
                     "correct, and complete code snippet that precisely addresses the given task description. "
                     "Use the provided retrieved code snippets OR web search results as context, inspiration, or building blocks if they are relevant. "
                     "If the provided context is irrelevant or unhelpful, generate the code from scratch based on the task.\n\n"
                     "Task Description:\n"
                     "```\n{task_description}\n```\n\n"
                     "Provided Context (from database search or web search):\n" # 
                     "```\n{retrieved_code_snippets}\n```\n\n" 
                     "Desired programming language (if known, otherwise infer from task or snippets): {language_hint}\n\n"
                     "IMPORTANT INSTRUCTIONS:\n"
                     "1. Only output the raw code for the solution. Do NOT include any explanatory text, markdown formatting (like ```python), or apologies before or after the code block.\n"
                     "2. If the task implies a specific language (e.g., 'Python function', 'JavaScript class'), use that language.\n"
                     "3. Include docstrings or comments within the code where appropriate to explain its functionality.\n"
                     "4. Ensure the code is complete and directly addresses the task.\n\n"
                     "Refined Code:"
        )
        self.refine_chain = self.refine_prompt | self.refiner_llm | self.refine_output_parser

    # --- Service Methods ---

    def evaluate_relevance(self, query: str, document: str) -> float:
        """Evaluates relevance using configured helper LLM."""
        logger.debug(f"Evaluating relevance for query: '{query[:50]}...' and doc: '{document[:50]}...'")
        try:
            result = self.relevance_chain.invoke({"query": query, "document": document})
            logger.debug(f"Relevance score: {result['score']:.2f}, Reasoning: {result.get('reasoning', 'N/A')}")
            return result.get('score', 0.0) # Default to 0 if 'score' somehow missing
        except Exception as e:
            logger.error(f"Error during relevance evaluation: {e}", exc_info=True)
            return 0.0 # Assume not relevant on error

    def rewrite_web_query(self, query: str) -> str:
        """Rewrites query for web search using configured helper LLM."""
        logger.debug(f"Rewriting query for web search: '{query[:50]}...'")
        try:
            result = self.rewrite_chain.invoke({"query": query})
            rewritten = result.get('rewritten_query', query) # Fallback to original if parsing fails
            logger.debug(f"Rewritten query: {rewritten}")
            return rewritten
        except Exception as e:
            logger.error(f"Error during query rewriting: {e}", exc_info=True)
            return query # Fallback to original query on error

    def extract_key_info(self, query: str, document: str) -> List[str]:
        """Extracts key points from text using configured helper LLM."""
        logger.debug(f"Extracting key info from document (context: '{query[:50]}...'): '{document[:100]}...'")
        try:
            result = self.extract_chain.invoke({"query": query, "document": document})
            key_points = result.get('key_points', [])
            logger.debug(f"Extracted {len(key_points)} key points.")
            return key_points
        except Exception as e:
            logger.error(f"Error during key info extraction: {e}", exc_info=True)
            return [f"Error during extraction: {e}"] # Return error message as a point

    def perform_web_search(self, original_query: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Performs web search, extracts info, and returns knowledge + sources."""
        logger.info(f"Performing web search for query: '{original_query[:50]}...'")
        rewritten_query = self.rewrite_web_query(original_query)
        try:
            web_results_str = self.web_search_tool.run(rewritten_query)
            if not web_results_str:
                 logger.warning(f"Web search for '{rewritten_query}' returned empty results.")
                 return ["No information found from web search."], []

            sources = parse_search_results(web_results_str)
            # Extract key info from the raw result string (often contains snippets)
            extracted_knowledge = self.extract_key_info(query=original_query, document=web_results_str)

            logger.info(f"Web search successful. Found {len(sources)} potential sources, extracted {len(extracted_knowledge)} key points.")
            return extracted_knowledge, sources

        except Exception as e:
            logger.error(f"Error during web search execution or processing: {e}", exc_info=True)
            return [f"Error during web search: {e}"], []

    def refine_code(
        self,
        task_description: str,
        context_str: str, # Combined context from retrieval/web
        language_hint: Optional[str] = "python"
    ) -> str:
        """Refines code using the main context string and configured refiner LLM."""
        logger.info(f"Refining code for task: '{task_description[:100]}...' using provided context.")
        if not context_str:
            context_str = "No context provided. Generate from scratch based on the task description."

        try:
            refined_code = self.refine_chain.invoke({
                "task_description": task_description,
                "retrieved_code_snippets": context_str, # Feed combined context here
                "language_hint": language_hint or "Try to infer from task or context"
            })
            # Basic cleaning remains the same
            cleaned_code = refined_code.strip().split('</think>')[-1]
            # More robust cleaning for ``` variations
            if cleaned_code.startswith("```") and cleaned_code.endswith("```"):
                lines = cleaned_code.splitlines()
                if len(lines) >= 2:
                     # Remove first ``` marker line and last ``` line
                     cleaned_code = "\n".join(lines[1:-1]).strip()
                else: # Handle edge case like just ```python\n```
                    cleaned_code = ""
            elif cleaned_code.startswith("```"): # Handle case where end ``` is missing
                lines = cleaned_code.splitlines()
                if len(lines) >= 1:
                     cleaned_code = "\n".join(lines[1:]).strip()

            logger.debug(f"Refiner LLM generated code: {cleaned_code[:200]}...")
            return cleaned_code.replace('```', '').replace('```python', '')
        except Exception as e:
            logger.error(f"Error during LLM code refinement call: {e}", exc_info=True)
            return f"// Error during code refinement process: {e}"
