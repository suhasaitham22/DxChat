import streamlit as st
import ollama # Use the Ollama client library
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Symptom Helper (Informational Only)",
    page_icon="🧑‍⚕️",
    layout="wide"
)

# --- Critical Disclaimer ---
st.title("AI Symptom Helper 🧑‍⚕️")
st.warning(
    """
    ⚠️ **Disclaimer:** I am an AI assistant, not a medical professional.
    This tool is for **informational purposes only** and **cannot provide a diagnosis**.
    The information provided may not be accurate or complete.
    **Always consult a qualified healthcare provider** for any health concerns or before making any decisions related to your health or treatment.
    Do not disregard professional medical advice or delay seeking it because of something you have read or received from this tool.
    **If you think you may have a medical emergency, call your doctor or emergency services immediately.**
    """
)
st.divider()

# --- Model Selection (Optional but good for demo) ---
# You could list models Ollama has downloaded or offer choices
# For simplicity, we'll hardcode one for now.
LLM_MODEL = "mistral" # Or "llama3", "phi3" - make sure it's pulled via `ollama pull <model_name>`

# --- System Prompt (Crucial for Safety & Role Setting) ---
SYSTEM_PROMPT = """
You are an AI assistant designed to provide general information related to health symptoms based on user descriptions.
You are NOT a doctor or a diagnostic tool.
Your primary goal is to help users articulate their symptoms and provide *potential* related areas or conditions for them to discuss with a healthcare professional.

RULES:
1.  **NEVER provide a diagnosis.** Do not say "you might have X" or "it sounds like Y".
2.  **ALWAYS preface information** by stating it is general information and not medical advice.
3.  **ALWAYS strongly recommend** the user consult a qualified healthcare professional for accurate diagnosis and treatment.
4.  **If the user describes severe symptoms** (e.g., chest pain, difficulty breathing, severe bleeding, loss of consciousness), immediately and primarily advise them to seek emergency medical attention.
5.  **Keep responses concise and informative.** Focus on potential areas related to the described symptoms.
6.  **Do not ask for Personally Identifiable Information (PII).**
7.  **Remind the user of your limitations** (AI, not a doctor) frequently, perhaps in every response.
8.  **If asked for treatment advice, refuse** and reiterate the need to see a doctor.
"""

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
         {"role": "system", "content": SYSTEM_PROMPT}, # Add system prompt
         {"role": "assistant", "content": "Hello! How can I help you describe your symptoms today? Remember, I cannot provide medical advice or diagnosis. Please consult a healthcare professional for any health concerns."}
    ]

# --- Display Chat Messages ---
for message in st.session_state.messages:
    # Don't display the system prompt to the user in the chat interface
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Describe your symptoms... (e.g., 'I have a headache and feel tired')"):
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using Ollama
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        full_response = ""
        try:
            # Prepare messages for Ollama (it needs the full history)
            # Ensure system prompt is included if needed by the model/library (Ollama usually handles it implicitly if set in the model file or via API)
            # The 'ollama-python' library sends the list directly.
            stream = ollama.chat(
                model=LLM_MODEL,
                messages=st.session_state.messages, # Send the whole history
                stream=True,
            )

            # Stream the response
            for chunk in stream:
                full_response += chunk['message']['content']
                message_placeholder.markdown(full_response + "▌") # Add blinking cursor
                time.sleep(0.01) # Small delay for streaming effect

            message_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            full_response = "Sorry, I encountered an error. Please try again. Remember to consult a doctor for medical advice."
            message_placeholder.markdown(full_response)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Sidebar for Clear Chat & More Disclaimers ---
with st.sidebar:
    st.header("⚠️ Important Notes")
    st.error(
        """
        **This is an AI simulation.** It is NOT a substitute for professional medical advice, diagnosis, or treatment.
        Information may be inaccurate. **Consult a real doctor.**
        """
    )
    if st.button("Clear Chat History"):
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Chat history cleared. How can I help? Remember my limitations."}
        ]
        st.rerun() # Use st.rerun instead of experimental_rerun

    st.markdown("---")
    st.markdown(f"**Model:** `{LLM_MODEL}` (Running via Ollama)")
    st.markdown("**Developer:** [Your Name/Company Name]") # For presentation
