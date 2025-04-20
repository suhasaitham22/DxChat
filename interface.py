import ollama
import logging
from typing import List, Dict, Iterator, Any

logger = logging.getLogger(__name__)

class LLMError(Exception):
    """Custom exception for LLM interaction errors."""
    pass

def check_ollama_connection(model_name: str) -> bool:
    """Checks if Ollama is running and the specified model is available."""
    try:
        models = ollama.list()
        available_models = [m['name'].split(':')[0] for m in models.get('models', [])] # Get base model names
        # Check if the requested model name (or its base version) is available
        base_model_name = model_name.split(':')[0]
        if model_name in available_models or base_model_name in available_models:
             logger.info(f"Ollama connection successful. Model '{model_name}' or its base '{base_model_name}' appears available.")
             return True
        else:
            logger.warning(f"Ollama connection successful, but model '{model_name}' not found in available models: {available_models}")
            return False # Model specified not found
    except Exception as e:
        logger.error(f"Failed to connect to Ollama or list models: {e}", exc_info=True)
        return False # Connection failed

def get_available_ollama_models() -> List[str]:
    """Gets a list of available model names from Ollama."""
    try:
        models_info = ollama.list()
        # Return full model names (e.g., "mistral:latest")
        return [m['name'] for m in models_info.get('models', [])]
    except Exception as e:
        logger.error(f"Could not retrieve models from Ollama: {e}")
        return []

def get_llm_response_stream(model_name: str, messages: List[Dict[str, str]], options: Dict[str, Any] = None) -> Iterator[str]:
    """
    Gets a streamed response from the local LLM via Ollama.

    Args:
        model_name: The name of the Ollama model to use (e.g., "mistral:latest").
        messages: The list of messages (conversation history including system prompt).
        options: Optional dictionary of Ollama parameters (e.g., temperature).

    Yields:
        String chunks of the response content.

    Raises:
        LLMError: If there's an issue communicating with Ollama or streaming.
    """
    logger.info(f"Requesting streaming response from Ollama model: {model_name}")
    logger.debug(f"Messages sent to LLM: {messages}") # Be careful logging full messages in production
    if options:
        logger.debug(f"LLM Options: {options}")

    try:
        # Ensure messages is a list of dictionaries with 'role' and 'content'
        if not isinstance(messages, list) or not all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
             raise ValueError("Invalid format for 'messages'. Expected List[Dict[str, str]].")

        stream = ollama.chat(
            model=model_name,
            messages=messages,
            stream=True,
            options=options # Pass optional parameters
        )

        # Stream the response chunks
        for chunk in stream:
            if chunk and 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
            elif chunk and 'done' in chunk and chunk['done'] and 'error' in chunk:
                 # Handle errors reported within the stream by Ollama
                 error_message = chunk.get('error', 'Unknown streaming error')
                 logger.error(f"Ollama stream returned an error: {error_message}")
                 raise LLMError(f"LLM streaming error: {error_message}")

    except ollama.ResponseError as e:
        logger.error(f"Ollama API Response Error: {e.status_code} - {e.error}", exc_info=True)
        if "model not found" in str(e.error).lower():
             raise LLMError(f"Model '{model_name}' not found by Ollama. Ensure it is pulled and running.") from e
        else:
             raise LLMError(f"Ollama API error: {e.error}") from e
    except Exception as e:
        logger.error(f"Generic error during LLM communication: {e}", exc_info=True)
        raise LLMError(f"Failed to get response from LLM: {e}") from e
