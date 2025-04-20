import streamlit as st
import logging
import time
from pathlib import Path

# --- Project Structure Setup ---
# Ensure local modules can be imported
import sys
sys.path.append(str(Path(__file__).parent))

# --- Local Imports ---
import utils
import llm_interface as llm

# --- Configuration & Logging ---
# Load configuration early
CONFIG = utils.load_config()
# Set up logging based on config
logger = utils.setup_logging(CONFIG)

# --- Constants from Config ---
APP_CONFIG = CONFIG.get('app', {})
LLM_CONFIG = CONFIG.get('llm', {})
PROMPTS_CONFIG = CONFIG.get('prompts', {})

PAGE_TITLE = APP_CONFIG.get('title', "AI Symptom Helper")
PAGE_ICON = APP_CONFIG.get('page_icon', "🧑‍⚕️")
MENU_ITEMS = APP_CONFIG.get('menu_items', {})

DEFAULT_MODEL = LLM_CONFIG.get('default_model', 'mistral')
FALLBACK_MODEL = LLM_CONFIG.get('fallback_model', None)
LLM_OPTIONS = {k: v for k, v in LLM_CONFIG.items() if k not in ['default_model', 'fallback_model']} # Extract options like temp, top_p

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

# --- State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, # Keep system prompt internally
        {"role": "assistant", "content": INITIAL_MSG}
    ]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODEL
if "ollama_available" not in st.session_state:
    # Check connection status once on startup/refresh
    st.session_state.ollama_available = llm.check_ollama_connection(DEFAULT_MODEL)
    if not st.session_state.ollama_available and FALLBACK_MODEL:
        logger.warning(f"Default model '{DEFAULT_MODEL}' check failed. Trying fallback '{FALLBACK_MODEL}'.")
        st.session_state.ollama_available = llm.check_ollama_connection(FALLBACK_MODEL)
        if st.session_state.ollama_available:
            st.session_state.selected_model = FALLBACK_MODEL
            logger.info(f"Using fallback model: {FALLBACK_MODEL}")
        else:
             logger.error("Ollama connection failed for both default and fallback models.")
    elif not st.session_state.ollama_available:
         logger.error(f"Ollama connection failed for default model '{DEFAULT_MODEL}' and no fallback specified.")

# --- Sidebar ---
with st.sidebar:
    st.title("Settings & Info")
    st.divider()

    # Model Selection
    st.subheader("LLM Configuration")
    available_models = llm.get_available_ollama_models()

    if not st.session_state.ollama_available:
         st.error("🔴 Ollama connection failed. Please ensure Ollama is running and accessible.")
         # Optionally disable model selection if connection failed
         model_options = [st.session_state.selected_model] # Show current (potentially unavailable)
         st.info(f"Attempting to use: `{st.session_state.selected_model}`. Functionality may be limited.")

    elif not available_models:
         st.warning("Could not fetch models from Ollama, but connection seems okay. Using default.")
         model_options = [st.session_state.selected_model]
    else:
         # If current model isn't in list (e.g., was default but not pulled), add it
         if st.session_state.selected_model not in available_models:
             model_options = [st.session_state.selected_model] + available_models
         else:
             model_options = available_models

         # Find index of currently selected model
         try:
            current_index = model_options.index(st.session_state.selected_model)
         except ValueError:
            current_index = 0 # Default to first if not found somehow
            st.session_state.selected_model = model_options[0]

         selected = st.selectbox(
             "Select LLM Model:",
             options=model_options,
             index=current_index,
             help="Choose the local LLM to interact with. Make sure it's pulled in Ollama."
         )
         if selected != st.session_state.selected_model:
             logger.info(f"User selected model: {selected}")
             # Check if the newly selected model is actually running/available
             if llm.check_ollama_connection(selected):
                 st.session_state.selected_model = selected
                 st.success(f"Switched to model: `{selected}`")
                 # Optional: Clear chat history on model switch? Or keep it?
                 # st.session_state.messages = [...] # Reset if desired
                 # st.rerun()
             else:
                 st.error(f"Could not verify newly selected model '{selected}'. Keeping previous selection.")


    st.caption(f"Using: `{st.session_state.selected_model}`")

    # Clear Chat Button
    st.divider()
    if st.button("Clear Chat History", key="clear_chat"):
        logger.info("Chat history cleared by user.")
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": INITIAL_MSG}
        ]
        st.rerun() # Use st.rerun for modern Streamlit

    # Important Notes / Disclaimer
    st.divider()
    st.error(DISCLAIMER_SHORT) # Short, always visible disclaimer
    with st.expander("⚠️ Show Full Disclaimer & Important Notes"):
        st.warning(DISCLAIMER_LONG)
        st.markdown("---")
        st.markdown(f"**Model:** `{st.session_state.selected_model}`")
        st.markdown("**Status:** " + ("🟢 Connected" if st.session_state.ollama_available else "🔴 Disconnected"))
        st.markdown("**Developer:** [Your Name / Company Name Here]") # For presentation


# --- Main Chat Interface ---
st.title(PAGE_TITLE)
st.warning(DISCLAIMER_LONG, icon="⚠️") # Prominent main disclaimer

# Display chat messages (excluding system prompt)
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Describe symptoms here... (e.g., 'headache and fatigue')"):
    logger.info(f"User input received: {prompt[:50]}...") # Log snippet
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
            # Ensure Ollama is available before attempting to stream
            if not st.session_state.ollama_available:
                 raise llm.LLMError("Ollama connection is not available. Cannot process request.")

            # Start streaming response from LLM
            response_stream = llm.get_llm_response_stream(
                model_name=st.session_state.selected_model,
                messages=[m for m in st.session_state.messages if m['role'] != 'system'], # Send history excluding system prompt if Ollama handles it implicitly
                # messages=st.session_state.messages, # Or send full history if model needs explicit system prompt in context
                options=LLM_OPTIONS
            )

            # Stream response to the UI
            for chunk in response_stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌") # Simulate typing cursor
                time.sleep(0.01) # Small delay for smoother streaming appearance

            message_placeholder.markdown(full_response) # Final response display

        except llm.LLMError as e:
            logger.error(f"LLMError encountered: {e}", exc_info=True)
            full_response = f"{ERROR_MSG}\n\n**Error details (for debugging):** {e}"
            message_placeholder.error(full_response) # Display error in chat
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            full_response = ERROR_MSG
            message_placeholder.error(full_response)

    # Add final assistant response (or error message) to state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
