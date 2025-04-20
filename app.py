import streamlit as st
import logging
import time
from pathlib import Path

# --- Project Structure Setup ---
import sys
sys.path.append(str(Path(__file__).parent))

# --- Local Imports ---
import utils
import llm_interface as llm # Renamed for clarity, now uses HF

# --- Configuration & Logging ---
CONFIG = utils.load_config()
logger = utils.setup_logging(CONFIG)

# --- Constants from Config ---
APP_CONFIG = CONFIG.get('app', {})
LLM_CONFIG = CONFIG.get('llm', {})
PROMPTS_CONFIG = CONFIG.get('prompts', {})

PAGE_TITLE = APP_CONFIG.get('title', "AI Symptom Helper")
PAGE_ICON = APP_CONFIG.get('page_icon', "🧑‍⚕️")
MENU_ITEMS = APP_CONFIG.get('menu_items', {})

DEFAULT_MODEL = LLM_CONFIG.get('default_model', "mistralai/Mistral-7B-Instruct-v0.1")
FALLBACK_MODEL = LLM_CONFIG.get('fallback_model', None)
# Extract LLM parameters directly, filter out model names
LLM_PARAMS = {k: v for k, v in LLM_CONFIG.items() if k not in ['default_model', 'fallback_model']}

SYSTEM_PROMPT = PROMPTS_CONFIG.get('system_prompt', "You are a helpful AI assistant.")
INITIAL_MSG = PROMPTS_CONFIG.get('initial_assistant_message', "How can I help?")
ERROR_MSG = PROMPTS_CONFIG.get('error_message', "Sorry, an error occurred.")
DISCLAIMER_SHORT = PROMPTS_CONFIG.get('disclaimer_short', "Not medical advice.")
DISCLAIMER_LONG = PROMPTS_CONFIG.get('disclaimer_long', "This is not medical advice. Consult a doctor.")

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    menu_items=MENU_ITEMS
)

# --- Initialize HF Clients ---
# Cache clients to avoid re-initialization on every interaction
@st.cache_resource
def get_clients():
    api_client = llm.get_hf_api_client()
    inference_client = llm.get_hf_inference_client()
    return api_client, inference_client

hf_api_client, hf_inference_client = get_clients()
hf_available = hf_inference_client is not None

# --- State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        # No system prompt here, handled separately in API call formatting
        {"role": "assistant", "content": INITIAL_MSG}
    ]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODEL
if "hf_available" not in st.session_state:
     st.session_state.hf_available = hf_available
     if not hf_available:
         logger.error("Hugging Face client initialization failed. Check token in secrets.")


# --- Sidebar ---
with st.sidebar:
    st.title("Settings & Info")
    st.divider()

    # Model Selection
    st.subheader("LLM Configuration")

    if not st.session_state.hf_available:
        st.error("🔴 Hugging Face client failed to initialize. Check API token in Streamlit secrets (`huggingface.token`).")
        model_options = [st.session_state.selected_model] # Show current config
        st.info(f"Configured model: `{st.session_state.selected_model}`. API connection failed.")
    else:
        # Fetch models only if client is available
        # Cache this potentially slow call
        @st.cache_data(ttl=3600) # Cache for 1 hour
        def cached_get_hf_models(_client): # Pass client to make cache key depend on it implicitly
            return llm.get_available_hf_models(hf_api_client) # Use the API client here

        available_models = cached_get_hf_models(hf_api_client)

        if not available_models:
            st.warning("Could not fetch models from Hugging Face Hub. Using configured default.")
            model_options = [st.session_state.selected_model]
            if FALLBACK_MODEL and st.session_state.selected_model != FALLBACK_MODEL:
                model_options.append(FALLBACK_MODEL)
        else:
            model_options = available_models
            # Ensure default and selected models are in the list for the dropdown
            if st.session_state.selected_model not in model_options:
                model_options.insert(0, st.session_state.selected_model)
            if DEFAULT_MODEL not in model_options:
                 model_options.insert(0, DEFAULT_MODEL)


        # Find index of currently selected model
        try:
            current_index = model_options.index(st.session_state.selected_model)
        except ValueError:
            st.warning(f"Selected model {st.session_state.selected_model} not in fetched list. Defaulting selection.")
            current_index = 0 # Default to first if not found
            st.session_state.selected_model = model_options[0]

        selected = st.selectbox(
            "Select LLM Model (from Hugging Face Hub):",
            options=list(set(model_options)), # Ensure unique options
            index=current_index,
            help="Choose an instruction-following model from Hugging Face Inference API."
        )
        if selected != st.session_state.selected_model:
            logger.info(f"User selected model: {selected}")
            # Optional: Check availability of the newly selected model? Can be slow.
            # if llm.check_hf_model_availability(selected, hf_inference_client):
            st.session_state.selected_model = selected
            st.success(f"Switched to model: `{selected}`")
            # else:
            #    st.error(f"Could not verify model '{selected}' availability via API. Keeping previous.")

    st.caption(f"Using: `{st.session_state.selected_model}`")

    # Clear Chat Button
    st.divider()
    if st.button("Clear Chat History", key="clear_chat"):
        logger.info("Chat history cleared by user.")
        st.session_state.messages = [
            {"role": "assistant", "content": INITIAL_MSG}
        ]
        st.rerun()

    # Important Notes / Disclaimer
    st.divider()
    st.error(DISCLAIMER_SHORT) # Short, always visible disclaimer
    with st.expander("⚠️ Show Full Disclaimer & Important Notes"):
        st.warning(DISCLAIMER_LONG)
        st.markdown("---")
        st.markdown(f"**Model:** `{st.session_state.selected_model}` (via HF Inference API)")
        st.markdown("**Status:** " + ("🟢 Connected to HF API" if st.session_state.hf_available else "🔴 HF API Connection Failed"))
        st.markdown("**Developer:** [Your Name / Company Name Here]") # For presentation


# --- Main Chat Interface ---
st.title(PAGE_TITLE)
st.warning(DISCLAIMER_LONG, icon="⚠️") # Prominent main disclaimer

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Describe symptoms here... (e.g., 'headache and fatigue')"):
    logger.info(f"User input received: {prompt[:50]}...")
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        full_response = ""
        try:
            # Ensure HF client is available
            if not st.session_state.hf_available or hf_inference_client is None:
                 raise llm.LLMError("Hugging Face Inference Client is unavailable. Cannot process request.")

            # Start streaming response from LLM
            response_stream = llm.get_hf_llm_response_stream(
                client=hf_inference_client,
                model_id=st.session_state.selected_model,
                messages=st.session_state.messages, # Pass full history for formatting
                system_prompt=SYSTEM_PROMPT, # Pass system prompt separately
                llm_params=LLM_PARAMS
            )

            # Stream response to the UI
            for chunk in response_stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌") # Simulate typing cursor
                # No time.sleep needed usually, streaming gives the effect

            message_placeholder.markdown(full_response) # Final response display

        except llm.LLMError as e:
            logger.error(f"LLMError encountered: {e}", exc_info=False) # Don't need full traceback for common LLM errors
            full_response = f"{ERROR_MSG}\n\n**Details:** {e}"
            message_placeholder.error(full_response) # Display error in chat
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            full_response = ERROR_MSG
            message_placeholder.error(full_response)

    # Add final assistant response (or error message) to state only if successful or known error
    # Avoid adding duplicate messages on unexpected crashes
    if full_response:
       st.session_state.messages.append({"role": "assistant", "content": full_response})
