import logging
from typing import List, Dict, Any, Optional

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings

logger = logging.getLogger(__name__)

class OllamaCodeRefiner:
    def __init__(
        self,
        model_name: str = settings.OLLAMA_CODE_MODEL,
        base_url: str = settings.OLLAMA_BASE_URL,
        temperature: float = 0.1, # Lower temp for more deterministic code
    ):
        try:
            self.llm = Ollama(model=model_name, base_url=base_url, temperature=temperature)
            # Test connection
            self.llm.invoke("Respond with only 'ACK' to confirm connection.")
            logger.info(f"Ollama model '{model_name}' connected at {base_url}.")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama model '{model_name}' at {base_url}: {e}", exc_info=True)
            raise ConnectionError(f"Ollama connection failed for model {model_name}") from e

        # This prompt needs careful crafting for your specific use case and LLM
        self.refine_prompt_template = PromptTemplate(
            input_variables=["task_description", "retrieved_code_snippets", "language_hint"],
            template="You are an expert AI programmer. Your goal is to generate a high-quality, "
                     "correct, and complete code snippet that precisely addresses the given task description. "
                     "Use the provided retrieved code snippets as context, inspiration, or building blocks if they are relevant. "
                     "If the snippets are irrelevant or unhelpful, generate the code from scratch based on the task.\n\n"
                     "Task Description:\n"
                     "```\n{task_description}\n```\n\n"
                     "Retrieved Code Snippets (Context):\n"
                     "```\n{retrieved_code_snippets}\n```\n\n"
                     "Desired programming language (if known, otherwise infer from task or snippets): {language_hint}\n\n"
                     "IMPORTANT INSTRUCTIONS:\n"
                     "1. Only output the raw code for the solution. Do NOT include any explanatory text, markdown formatting (like ```python), or apologies before or after the code block.\n"
                     "2. If the task implies a specific language (e.g., 'Python function', 'JavaScript class'), use that language.\n"
                     "3. Include docstrings or comments within the code where appropriate to explain its functionality.\n"
                     "4. Ensure the code is complete and directly addresses the task.\n\n"
                     "Refined Code:"
        )
        self.output_parser = StrOutputParser()
        self.refine_chain = self.refine_prompt_template | self.llm | self.output_parser

    def refine_code(
        self,
        task_description: str,
        retrieved_snippets: List[Dict[str, Any]], # List of {"content": ..., "metadata": ...}
        language_hint: Optional[str] = "python" # Default or inferred language
    ) -> str:
        if not task_description:
            logger.warning("Task description is empty for code refinement.")
            return "// No task description provided."

        context_str = "No relevant code snippets found."
        if retrieved_snippets:
            formatted_snippets = []
            for i, snippet in enumerate(retrieved_snippets):
                meta_info = f"Source: {snippet['metadata'].get('source_id', 'unknown')}, chunk: {snippet['metadata'].get('chunk_index', 'N/A')}"
                if 'description' in snippet['metadata']: # If you store descriptions
                    meta_info += f"\nDescription: {snippet['metadata']['description']}"
                formatted_snippets.append(f"-- Snippet {i+1} --\n{meta_info}\nCode:\n{snippet['content']}\n-- End Snippet {i+1} --")
            context_str = "\n\n".join(formatted_snippets)

        logger.info(f"Refining code for task: '{task_description[:100]}...' with {len(retrieved_snippets)} context snippets.")
        try:
            refined_code = self.refine_chain.invoke({
                "task_description": task_description,
                "retrieved_code_snippets": context_str,
                "language_hint": language_hint or "Try to infer from task or context"
            })
            # Basic cleaning: some models might still add ```
            cleaned_code = refined_code.strip()
            if cleaned_code.startswith("```") and cleaned_code.endswith("```"):
                # Remove the first line (e.g., ```python) and the last line (```)
                lines = cleaned_code.splitlines()
                if len(lines) > 1:
                    cleaned_code = "\n".join(lines[1:-1])
                else: # Only ```python``` orsimilar
                    cleaned_code = "\n".join(lines).replace("```"+(language_hint or "python"), "").replace("```", "").strip()

            logger.debug(f"LLM generated code: {cleaned_code[:200]}...")
            return cleaned_code
        except Exception as e:
            logger.error(f"Error during LLM code refinement: {e}", exc_info=True)
            return f"// Error during code refinement: {e}"
