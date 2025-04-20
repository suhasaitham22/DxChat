import streamlit as st
import logging
from typing import List, Dict, Iterator, Any
from huggingface_hub import InferenceClient, HfApi
from huggingface_hub.hf_api import HfApi, ModelInfo
# Removed problematic import: from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, Token
import requests
import time

logger = logging.getLogger(__name__)

class LLMError(Exception):
    """Custom exception for LLM interaction errors."""
    pass

# --- Hugging Face API Configuration ---
# Get the token from Streamlit secrets
# This assumes secrets are loaded elsewhere (e.g., in app.py)
# We use st.secrets directly here for simplicity in this module context,
# but passing the client initialized with the token might be cleaner.
HF_TOKEN = st.secrets.get("huggingface", {}).get("token")

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
        # Depending on the HF model and usage, InferenceClient might sometimes
        # work without a token for public models, but explicit token usage
        # is required for reliability, private models, and higher rate limits.
        st.error("Hugging Face API Token is missing from Streamlit Secrets.", icon="🚨")
        return None
    try:
        return InferenceClient(token=HF_TOKEN)
    except Exception as e:
        logger.error(f"Failed to initialize Hugging Face Inference client: {e}", exc_info=True)
        return None

def check_hf_model_availability(model_id: str, client: InferenceClient | None) -> bool:
    """
    Checks if a model seems available via the Inference API (basic check).
    Note: This is a basic check and doesn't guarantee the model won't fail later.
    """
    if not client:
        logger.warning("Cannot check model availability: Inference Client not initialized.")
        return False
    try:
        # Send a minimal prompt to see if it returns without immediate error.
        client.text_generation(prompt=".", model=model_id, max_new_tokens=1)
        logger.info(f"Model '{model_id}' appears available via Inference API (basic check successful).")
        return True
    except Exception as e:
        # Catching broad exception as API errors can vary.
        logger.warning(f"Basic check failed for model '{model_id}' via Inference API: {e}")
        return False

def get_available_hf_models(client: HfApi | None, task_filter="text-generation") -> List[str]:
    """Gets a list of potentially suitable models from Hugging Face Hub."""
    if not client:
        logger.warning("Cannot fetch models: Hugging Face API client not initialized.")
        return []
    try:
        # Fetch models, filter by task. Limit to avoid overwhelming dropdown.
        models = client.list_models(
            filter=task_filter,
            sort="downloads",
            direction=-1,
            limit=50 # Adjust limit as needed
        )
        model_ids = [model.modelId for model in models]
        logger.info(f"Fetched {len(model_ids)} potential models from Hugging Face Hub.")
        return model_ids
    except Exception as e:
        logger.error(f"Could not retrieve models from Hugging Face Hub: {e}", exc_info=True)
        return []

def format_messages_for_hf(messages: List[Dict[str, str]], system_prompt: str | None = None) -> str:
    """
    Formats conversation history into a single string suitable for many Hugging Face
    instruction-tuned models (like Mistral, Zephyr, Llama-2-chat).
    Adjust this format based on the specific model's requirements if needed.
    """
    formatted_prompt = ""
    # Apply system prompt using a common instruction format if provided
    if system_prompt:
        # Example format for Mistral/Zephyr - adjust if needed for other models
        formatted_prompt += f"<|system|>\n{system_prompt}</s>\n"

    # Process user/assistant messages
    for msg in messages:
        if msg["role"] == "user":
            formatted_prompt += f"<|user|>\n{msg['content']}</s>\n"
        elif msg["role"] == "assistant":
            # Append assistant message for context
            formatted_prompt += f"<|assistant|>\n{msg['content']}</s>\n"
        # System messages are handled above or ignored if already processed

    # Add the final prompt marker for the assistant to start generating
    formatted_prompt += "<|assistant|>\n"
    # logger.debug(f"Formatted prompt for HF model:\n{formatted_prompt}") # Be cautious logging potentially sensitive data
    return formatted_prompt


def get_hf_llm_response_stream(
    client: InferenceClient,
    model_id: str,
    messages: List[Dict[str, str]],
    system_prompt: str | None,
    llm_params: Dict[str, Any]
) -> Iterator[str]:
    """
    Gets a streamed response from the Hugging Face Inference API using text_generation.

    Args:
        client: Initialized HuggingFace InferenceClient.
        model_id: The Hugging Face model ID (e.g., "mistralai/Mistral-7B-Instruct-v0.1").
        messages: Conversation history (user/assistant messages).
        system_prompt: The system prompt string, if any.
        llm_params: Dictionary of parameters like temperature, max_new_tokens, etc.

    Yields:
        String chunks of the response content.

    Raises:
        LLMError: If there's an issue communicating with the API or streaming.
    """
    if not client:
        raise LLMError("Hugging Face Inference Client is not initialized.")

    # Prepare the prompt string using the formatter
    # Exclude system messages from the list if handled separately by the formatter
    user_assistant_messages = [m for m in messages if m['role'] != 'system']
    prompt_text = format_messages_for_hf(user_assistant_messages, system_prompt)

    logger.info(f"Requesting streaming response from HF Inference API model: {model_id}")

    # Ensure necessary parameters are present, provide defaults if missing
    if 'max_new_tokens' not in llm_params:
        llm_params['max_new_tokens'] = 512 # Sensible default
    if 'temperature' not in llm_params:
        llm_params['temperature'] = 0.6 # Sensible default
    if 'top_p' not in llm_params:
        llm_params['top_p'] = 0.9 # Sensible default

    try:
        # Use the client's text_generation method with stream=True and details=False
        stream = client.text_generation(
            prompt=prompt_text,
            model=model_id,
            stream=True,
            details=False, # Ensure this is False to get string chunks
            **llm_params # Pass temperature, top_p, max_new_tokens etc.
        )

        # Process the stream of string chunks
        generated_text = ""
        for chunk in stream:
             # Since details=False, each item yielded should be a string chunk
             if isinstance(chunk, str):
                 generated_text += chunk
                 yield chunk
             else:
                 # Log if something unexpected is received, but try to continue
                 logger.warning(f"Received unexpected chunk type in stream (expected str with details=False): {type(chunk)}")
                 # Attempt to convert to string, might fail
                 try:
                     str_chunk = str(chunk)
                     generated_text += str_chunk
                     yield str_chunk
                 except Exception:
                    logger.warning(f"Could not convert unexpected chunk to string: {chunk}")


        logger.info(f"Finished streaming from {model_id}. Total length: {len(generated_text)}")
        if not generated_text.strip() and len(user_assistant_messages) > 0:
             logger.warning(f"Model {model_id} returned an empty response for the given prompt.")
             # Optional: yield a message indicating empty response? Or handle in app.py
             # yield "[Model returned an empty response]"


    except requests.exceptions.RequestException as e:
        logger.error(f"Network error connecting to Hugging Face API: {e}", exc_info=True)
        raise LLMError(f"Network error: Could not connect to Hugging Face API. {e}") from e
    except Exception as e:
        # Catch specific HF API errors if possible, otherwise generic
        error_message = str(e)
        logger.error(f"Error during Hugging Face API call or streaming: {error_message}", exc_info=True)

        # Check for common errors based on string content (may need refinement)
        if "Rate limit reached" in error_message:
            raise LLMError("Rate limit reached on Hugging Face API. Please try again later or check your HF plan.") from e
        elif "Model" in error_message and ("is currently loading" in error_message or "State: Loading" in error_message):
             # Inform the user the model is loading
             raise LLMError("The selected model is currently loading on the server. This can take some time, especially on the free tier. Please wait a minute and try again.") from e
        elif "not found" in error_message or "does not exist" in error_message:
             raise LLMError(f"Model '{model_id}' not found or not accessible via the Inference API.") from e
        elif "Input validation error" in error_message:
             raise LLMError(f"Input validation error. Check prompt formatting or parameters. Details: {error_message}") from e
        elif "GPU availability" in error_message: # Specific error example if GPUs are unavailable
             raise LLMError("The Inference API is currently experiencing high load or resource constraints. Please try again later.") from e
        else:
             # Generic error for other cases
             raise LLMError(f"Failed to get response from Hugging Face API: {e}") from e
