# DxChat


# AI Symptom Helper Chatbot 🧑‍⚕️

A Streamlit web application demonstrating an AI chatbot designed to provide **general information** about health symptoms. This tool uses open-source Large Language Models (LLMs) accessed via the Hugging Face Inference API.

**➡️ Live Demo :** [[Link to your Streamlit Cloud App](https://dxchat-paakwi7x3w7tf36v7h26xq.streamlit.app/)]

---

## ⚠️ **EXTREMELY IMPORTANT DISCLAIMER** ⚠️

*   **This is NOT a medical device or diagnostic tool.**
*   **This application is for informational and demonstration purposes ONLY.**
*   The AI can make mistakes, provide inaccurate, incomplete, or biased information. **It does NOT replace professional medical advice.**
*   **NEVER use this tool to self-diagnose or make decisions about your health or treatment.**
*   **ALWAYS consult a qualified healthcare provider** for any health concerns, diagnosis, or treatment.
*   **If you think you have a medical emergency, call your doctor or emergency services IMMEDIATELY.**
*   Use of this tool is at your own risk. The developers assume no liability for any actions taken based on the information provided by this application.

---

## Features

*   **Conversational Interface:** Chat with an AI about health symptoms.
*   **Symptom Information:** Receive general information potentially related to described symptoms (NOT a diagnosis).
*   **Open-Source LLM Powered:** Utilizes models hosted on the Hugging Face Hub via their Inference API.
*   **Configurable:** Easily change the underlying LLM model and system prompts via `config.yaml`.
*   **Safety Focused:** Strong system prompts and disclaimers designed to prevent diagnosis and encourage professional consultation.
*   **Streamlit Cloud Ready:** Designed for easy deployment on Streamlit Community Cloud.

## Technology Stack

*   **Python:** Core programming language.
*   **Streamlit:** Web application framework.
*   **Hugging Face Inference API:** Service to run LLMs remotely.
*   **huggingface_hub:** Python client library for interacting with Hugging Face services.
*   **PyYAML:** For handling the `config.yaml` configuration file.

## Setup and Installation (Local Development)

1.  **Clone the Repository:**
    ```bash
    git clone [(https://github.com/suhasaitham22/DxChat/tree/main)]
    cd [repository-folder-name]
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    # Activate the environment
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Hugging Face API Token:**
    *   You need a Hugging Face account: [huggingface.co](https://huggingface.co)
    *   Generate an API token with **`read`** permissions: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
    *   Create a file named `secrets.toml` inside a `.streamlit` directory: `.streamlit/secrets.toml`
    *   Add your token to the `secrets.toml` file:
        ```toml
        # .streamlit/secrets.toml
        [huggingface]
        token = "hf_YOUR_HUGGINGFACE_READ_TOKEN"
        ```
    *   **(Important):** Add `.streamlit/secrets.toml` to your `.gitignore` file to avoid committing your secret token!

## Running Locally

Once the setup is complete, run the Streamlit application:

```bash
streamlit run app.py
