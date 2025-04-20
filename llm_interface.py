import streamlit as st
import logging
from typing import List, Dict, Iterator, Any
from huggingface_hub import InferenceClient, HfApi
from huggingface_hub.hf_api import HfApi, ModelInfo
# from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, Token
import requests
import time

logger = logging.getLogger(__name__)

class LLMError(Exception):
    """Custom exception for LLM interaction errors."""
    pass

# --- Hugging Face API Configuration ---
# Get the token from Streamlit secrets
HF_TOKEN = st.secrets.get("huggingface", {}).get("token")
# Define common parameters from config (loaded elsewhere or passed in)
# Example: LLM_CONFIG = utils.CONFIG.get('llm', {})

def get_hf_api_client() -> HfApi | None:
    """Initializes and returns a Hugging Face API client if token is available."""
    if not HF_TOKEN:
        logger.warning("Hugging Face API token not found in Streamlit secrets.")
        return None
    try:
        return HfApi(token=HF_TOKEN)
    except Exception as e:
        logger.error(f"Failed to initialize Hugging Face API client: {e}", exc_info=True)
        return None

def get_hf_inference_client() -> InferenceClient | None:
    """Initializes and returns a Hugging Face Inference client if token is available."""
    if not HF_TOKEN:
        logger.warning("Hugging Face API token not found for Inference Client.")
        # InferenceClient can sometimes work without a token for public models,
        # but it's better practice and needed for private/gated models or higher rate limits.
        # For stability, let's require it.
        return None
    try:
        return InferenceClient(token=HF_TOKEN)
    except Exception as e:
        logger.error(f"Failed to initialize Hugging Face Inference client: {e}", exc_info=True)
        return None

def check_hf_model_availability(model_id: str, client: InferenceClient | None) -> bool:
    """Checks if a model seems available via the Inference API (basic check)."""
    if not client:
        logger.warning("Cannot check model availability: Inference Client not initialized.")
        return False
    try:
        # A simple way to check is to send a very short prompt.
        # This isn't foolproof, as the model might load slowly or fail later.
        # A better check might involve HfApi if more detail is needed, but InferenceClient ping is simpler.
        client.text_generation(prompt=".", model=model_id, max_new_tokens=1)
        logger.info(f"Model '{model_id}' appears available via Inference API.")
        return True
    except Exception as e:
        # Catching broad exception as API errors can vary (404, 5xx, timeout etc.)
        logger.warning(f"Model '{model_id}' check failed or model unavailable via Inference API: {e}")
        return False

def get_available_hf_models(client: HfApi | None, task_filter="text-generation") -> List[str]:
    """Gets a list of potentially suitable models from Hugging Face Hub."""
    if not client:
        logger.warning("Cannot fetch models: Hugging Face API client not initialized.")
        return []
    try:
        # Fetch models, filter by task. This list can be HUGE, so maybe limit it.
        models = client.list_models(
            filter=task_filter,
            sort="downloads", # Sort by downloads as a proxy for popularity/utility
            direction=-1,
            limit=50 # Limit the number of models fetched for the dropdown
            )
        model_ids = [model.modelId for model in models]
        logger.info(f"Fetched {len(model_ids)} potential models from Hugging Face Hub.")
        return model_ids
    except Exception as e:
        logger.error(f"Could not retrieve models from Hugging Face Hub: {e}", exc_info=True)
        return []

def format_messages_for_hf(messages: List[Dict[str, str]], system_prompt: str | None = None) -> str:
    """
    Formats conversation history into a single string suitable for many Hugging Face instruction-tuned models.
    Adjust this based on the specific model's expected prompt format (e.g., ChatML, Llama-2 chat, etc.).
    This is a generic example; specific models might need more tailored formatting.
    """
    formatted_prompt = ""
    # Prepend system prompt if provided and needed (some models use it differently)
    if system_prompt:
        # This format might work for Mistral/Zephyr style models
         formatted_prompt += f"<|system|>\n{system_prompt}</s>\n"

    # Process user/assistant messages
    for msg in messages:
        if msg["role"] == "user":
            formatted_prompt += f"<|user|>\n{msg['content']}</s>\n"
        elif msg["role"] == "assistant":
            formatted_prompt += f"<|assistant|>\n{msg['content']}</s>\n"
        # Ignore system messages here if handled above

    # Add the final prompt for the assistant to respond
    formatted_prompt += "<|assistant|>\n"
    logger.debug(f"Formatted prompt for HF model:\n{formatted_prompt}") # Careful logging potentially sensitive data
    return formatted_prompt


def get_hf_llm_response_stream(
    client: InferenceClient,
    model_id: str,
    messages: List[Dict[str, str]],
    system_prompt: str | None,
    llm_params: Dict[str, Any]
) -> Iterator[str]:
    """
    Gets a streamed response from the Hugging Face Inference API.

    Args:
        client: Initialized HuggingFace InferenceClient.
        model_id: The Hugging Face model ID (e.g., "mistralai/Mistral-7B-Instruct-v0.1").
        messages: Conversation history (excluding system prompt if handled separately).
        system_prompt: The system prompt string.
        llm_params: Dictionary of parameters like temperature, max_new_tokens, etc.

    Yields:
        String chunks of the response content.

    Raises:
        LLMError: If there's an issue communicating with the API or streaming.
    """
    if not client:
        raise LLMError("Hugging Face Inference Client is not initialized.")

    # Prepare the prompt string using the formatter
    # Exclude the system message from the list if it's passed separately
    user_assistant_messages = [m for m in messages if m['role'] != 'system']
    prompt_text = format_messages_for_hf(user_assistant_messages, system_prompt)

    logger.info(f"Requesting streaming response from HF Inference API model: {model_id}")

    # Ensure necessary parameters are present
    if 'max_new_tokens' not in llm_params:
        llm_params['max_new_tokens'] = 512 # Default if not set

    try:
        stream = client.text_generation(
            prompt=prompt_text,
            model=model_id,
            stream=True,
            details=False, # Don't need token details usually
            **llm_params # Pass temperature, top_p, max_new_tokens etc.
        )

        # Stream the response chunks
            generated_text = ""
            for chunk in stream:
                 # Directly process the yielded string chunk
                 if isinstance(chunk, str):
                     generated_text += chunk
                     yield chunk
                 else:
                     # Log if something unexpected is received, but don't crash
                     logger.warning(f"Received unexpected chunk type in stream (expected str with details=False): {type(chunk)}")

        logger.info(f"Finished streaming from {model_id}. Total length: {len(generated_text)}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error connecting to Hugging Face API: {e}", exc_info=True)
        raise LLMError(f"Network error: Could not connect to Hugging Face API. {e}") from e
    except Exception as e:
        # Catch specific HF API errors if possible, otherwise generic
        error_message = str(e)
        logger.error(f"Error during Hugging Face API call or streaming: {error_message}", exc_info=True)
        # Check for common errors
        if "Rate limit reached" in error_message:
            raise LLMError("Rate limit reached on Hugging Face API. Please try again later or upgrade your plan.") from e
        elif "Model" in error_message and "is currently loading" in error_message:
             # Wait a bit and tell user model is loading
             time.sleep(5) # Give it a few seconds
             raise LLMError("The selected model is loading on the server. Please wait a moment and try again.") from e
        elif "not found" in error_message or "does not exist" in error_message:
             raise LLMError(f"Model '{model_id}' not found or accessible via the Inference API.") from e
        else:
             raise LLMError(f"Failed to get response from Hugging Face API: {e}") from e
