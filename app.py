import streamlit as st
import time
import uuid
import os
# from queue import Queue # Removed Queue - not needed for Streamlit session handling
from transformers import AutoModelForCausalLM, AutoTokenizer # Removed TextStreamer for simplicity for now
import torch

# === MODEL LOADING === #

# Define available models with their Hugging Face paths
MODELS = {
    "Mistral 7B Instruct v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "Llama 2 7B Chat HF": "meta-llama/Llama-2-7b-chat-hf", # Renamed for clarity
    "Qwen 1.5 1.8B Chat": "Qwen/Qwen1.5-1.8B-Chat"       # Renamed for clarity
}

# Model and tokenizer cache (using Streamlit's caching)
@st.cache_resource # Use Streamlit's caching for models/tokenizers
def load_model(model_name):
    """Loads model and tokenizer from cache or disk."""
    model_path = MODELS.get(model_name)
    if not model_path:
        st.error(f"Model configuration not found for {model_name}")
        return None, None
    
    st.info(f"Loading model: {model_name}...") # Use info/progress
    try:
        # Ensure you have 'accelerate' installed via requirements.txt
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16, # Keep float16 for memory, consider float32 if issues arise
            device_map="auto" # Requires 'accelerate' library
            # low_cpu_mem_usage=True # Can help with large models if memory is tight during loading
        )
        st.success(f"Model loaded: {model_name}")
        return model, tokenizer
    except ImportError as e:
        st.error(f"Error loading model {model_name}: {e}. Make sure 'accelerate' is installed (check requirements.txt).")
        return None, None
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None, None

# === SESSION & HISTORY MANAGEMENT === #

MAX_HISTORY = 10  # Max chat history pairs (user + bot)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # List of lists: [[user_msg1, bot_msg1], [user_msg2, bot_msg2], ...]

if "current_model_name" not in st.session_state:
    st.session_state.current_model_name = list(MODELS.keys())[2] # Default to Qwen

# === HELPER FUNCTIONS === #

def generate_response(message, history, model_name):
    """Generate response using the selected model"""
    model, tokenizer = load_model(model_name) # Uses st.cache_resource
    if model is None or tokenizer is None:
        return "Error: Model not loaded.", history # Return error message

    # Format chat history for the specific model's template
    # Most models expect a list of dictionaries like:
    # [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    messages = []
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg and assistant_msg != "Typing...": # Don't include "Typing..." in history for model
            messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add the new user message
    messages.append({"role": "user", "content": message})

    try:
        # Apply the chat template
        # Note: Some older models might not have a default chat template configured.
        # You might need to manually format the prompt string if tokenizer.apply_chat_template fails.
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device) # Move input tensors to the same device as the model

        # --- Generation ---
        # Use a context manager to potentially handle device placement issues (optional)
        # with torch.inference_mode(): # Good practice for inference
        generation_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id # Set pad_token_id
        )

        # Decode only the newly generated tokens
        response_ids = generation_output[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        # Update history *after* successful generation
        # Ensure history doesn't exceed max length
        updated_history = history + [[message, response]]
        if len(updated_history) > MAX_HISTORY:
            updated_history = updated_history[-MAX_HISTORY:]

        return response, updated_history

    except Exception as e:
        st.error(f"Error during generation: {e}")
        return f"Error generating response: {e}", history # Return error and original history

def save_history(session_id, history):
    """Save chat history to a text file"""
    if not history:
        return None # Don't save empty history

    os.makedirs("chat_histories", exist_ok=True)
    filepath = os.path.join("chat_histories", f"chat_history_{session_id}.txt")
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            for user_input, bot_response in history:
                f.write(f"User: {user_input or ''}\nBot: {bot_response or ''}\n\n")
        return filepath
    except Exception as e:
        st.error(f"Failed to save history: {e}")
        return None

def clear_history():
    """Clear chat history in session state"""
    st.session_state.chat_history = []

def transcribe_audio(audio_file):
    """Simulate audio transcription (replace with actual transcription)"""
    # Placeholder: In a real app, you'd use a speech-to-text model/API
    # e.g., Hugging Face's 'openai/whisper-large-v3' or assemblyai, etc.
    st.warning("Audio transcription is currently simulated.")
    # Simulate reading the file to show it was received
    # audio_bytes = audio_file.read()
    # st.audio(audio_bytes)
    return f"Simulated transcription for: {audio_file.name}"

# === UI SETUP === #
st.set_page_config(layout="wide")
st.title("💬 Multi-Model AI Assistant")
st.markdown("Running on Hugging Face Spaces CPU - Model loading and inference might be slow.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")

    # Model selector
    selected_model_name = st.selectbox(
        "Choose Model",
        options=list(MODELS.keys()),
        key="current_model_name", # Link to session state key
        # index=list(MODELS.keys()).index(st.session_state.current_model_name) # Set index based on state
        help="Select the language model to chat with."
    )
    
    st.info(f"**Selected:** {st.session_state.current_model_name}")
    st.markdown(f"[Model Card]({MODELS.get(st.session_state.current_model_name,'')})", unsafe_allow_html=True)


    st.header("📜 Chat Controls")
    if st.button("Clear Chat History", key="clear_btn"):
        clear_history()
        st.success("Chat history cleared!")
        st.rerun() # Rerun to reflect the cleared history in the UI immediately

    # Download Button Logic
    history_filepath = save_history(st.session_state.session_id, st.session_state.chat_history)
    if history_filepath and os.path.exists(history_filepath):
        try:
            with open(history_filepath, "rb") as fp:
                st.download_button(
                    label="Download Chat History",
                    data=fp,
                    file_name=f"chat_history_{st.session_state.session_id}.txt",
                    mime="text/plain",
                    key="download_btn"
                )
        except Exception as e:
            st.error(f"Error preparing download: {e}")
    elif st.session_state.chat_history: # Only show button if there's history but saving failed maybe
         st.warning("Could not prepare history file for download.")


# --- Main Chat Area ---
st.header("Chat Window")

# Chatbot message display area
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        st.info("Starting a new chat session. Ask me anything!")
        
    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        if user_msg:
            with st.chat_message("user"):
                st.write(user_msg)
        if bot_msg:
            with st.chat_message("assistant"):
                st.write(bot_msg)


# Input area at the bottom
st.divider()
user_input = st.chat_input("Type your message or upload audio...")
# Note: st.chat_input provides a better UI than st.text_input + button for chat

# audio_bytes = None
# with st.expander("Upload Audio"):
#     voice_input = st.file_uploader("Upload an audio file (WAV, MP3)", type=["wav", "mp3", "m4a", "ogg"])
#     if voice_input:
#         audio_bytes = voice_input.getvalue()
#         st.audio(audio_bytes)

message_to_send = user_input
# if voice_input and not user_input: # Prioritize text input if both are provided
#     message_to_send = transcribe_audio(voice_input)

if message_to_send:
    # Add user message to history immediately for better UX
    st.session_state.chat_history.append([message_to_send, "Typing..."])
    
    # Rerun to show the user message and "Typing..." indicator
    st.rerun() 

# Separate block to handle generation after rerun shows user message
# Check if the last message is waiting for a response ("Typing...")
if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "Typing...":
    
    # Get the message that needs a response
    current_message = st.session_state.chat_history[-1][0]
    history_for_model = st.session_state.chat_history[:-1] # Don't send "Typing..." to model
    
    # Ensure history doesn't exceed max length before sending to model
    if len(history_for_model) > MAX_HISTORY:
         history_for_model = history_for_model[-MAX_HISTORY:]

    # Generate response
    # Use a spinner while generating
    with st.spinner(f"🤖 {st.session_state.current_model_name} is thinking..."):
        response, updated_history = generate_response(
            current_message,
            history_for_model, # Pass history *without* the latest user message/typing indicator
            st.session_state.current_model_name
        )

    # Update the last entry in session state history with the actual response
    st.session_state.chat_history[-1][1] = response
    
    # Rerun the script again to display the actual bot response
    st.rerun()