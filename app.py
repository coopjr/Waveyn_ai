# app.py (Final Version)

import gradio as gr
import time
import threading
import uuid
from queue import Queue

# === MODEL LOADING === #
# Models info
MODELS = {
    "Mistral 7B Q4": "mistral-model-path",
    "LLaMA 7B Q4": "llama-model-path",
    "Qwen 1.8B Q4 (Multilingual)": "qwen-model-path"
}

current_model_name = list(MODELS.keys())[0]
current_model_path = MODELS[current_model_name]

# Dummy load model function (Replace with actual loader)
def load_model(model_path):
    time.sleep(2)  # Simulate loading time
    return f"Loaded model from {model_path}"

model = load_model(current_model_path)

# === CHAT SESSION MANAGEMENT === #
sessions = {}

# === QUEUE SYSTEM === #
queue = Queue()

# === HISTORY LIMIT === #
MAX_HISTORY = 10

# === Helper functions === #
def generate_response(message, history, model_name):
    """Simulate model response"""
    for i in range(5):
        time.sleep(0.3)  # Simulate typing delay
        yield f"Generating reply{'.' * (i % 3 + 1)}"
    yield f"[{model_name}] Response to: {message}"


def save_history(session_id, history):
    with open(f"chat_history_{session_id}.txt", "w", encoding="utf-8") as f:
        for user_input, bot_response in history:
            f.write(f"User: {user_input}\nBot: {bot_response}\n\n")


def clear_history(session_id):
    sessions[session_id] = []

# === VOICE INPUT SUPPORT === #
def transcribe_audio(audio):
    return "Simulated transcription from audio."

# === PROGRESS BAR UPDATE (Async Simulation) === #
def async_progress(callback):
    for i in range(0, 101, 10):
        time.sleep(0.1)
        callback(i)

# === MAIN CHAT FUNCTION === #
def chat(message, history, session_id, model_name, progress=gr.Progress(track_tqdm=True)):
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    # Limit history for performance
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    # Simulate queue system
    queue.put(session_id)

    # Typing indicator simulation
    yield history + [[message, "Typing..."]]

    # Token streaming
    response = ""
    for partial in generate_response(message, history, model_name):
        response = partial
        yield history + [[message, response]]

    # Save final response
    history.append([message, response])
    sessions[session_id] = history

    # Save history to file
    save_history(session_id, history)

    # Done with queue
    queue.get()

    yield history

# === UI SETUP === #
theme_toggle = gr.themes.Default().set(
    body_background_fill_dark="*neutral_900",
    body_background_fill_light="*neutral_100"
)

with gr.Blocks(theme=theme_toggle) as demo:

    # Session ID
    session_id = str(uuid.uuid4())

    # Title
    gr.Markdown("# 🚀 Multi-Model AI Chatbot + Voice + History 📜")

    # Model Selector + Info Card
    with gr.Row():
        model_dropdown = gr.Dropdown(label="Choose Model", choices=list(MODELS.keys()), value=current_model_name)
        model_info = gr.Markdown(f"**Current Model:** {current_model_name}")

    # Chatbot
    chatbot = gr.Chatbot()

    # Voice + Text Input
    with gr.Row():
        txt_input = gr.Textbox(placeholder="Type your message...")
        voice_input = gr.Audio(source="microphone", type="filepath", label="Or speak")

    # Buttons
    with gr.Row():
        send_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear Chat")
        download_btn = gr.Button("Download Chat History")

    # Loading bar
    progress_bar = gr.Slider(0, 100, value=0, label="Loading Progress")

    # Functions
    def update_model(selected_model):
        global current_model_name, current_model_path, model
        current_model_name = selected_model
        current_model_path = MODELS[selected_model]
        model = load_model(current_model_path)
        return f"**Current Model:** {selected_model}"

    model_dropdown.change(fn=update_model, inputs=model_dropdown, outputs=model_info)

    def handle_send(message, audio, chat_history):
        if audio:
            message = transcribe_audio(audio)
        return chat(message, chat_history, session_id, current_model_name)

    send_btn.click(fn=handle_send, inputs=[txt_input, voice_input, chatbot], outputs=chatbot)

    def handle_clear():
        clear_history(session_id)
        return []

    clear_btn.click(fn=handle_clear, outputs=chatbot)s

    def handle_download():
        filepath = f"chat_history_{session_id}.txt"
        return gr.File.update(value=filepath, visible=True)

    download_btn.click(fn=handle_download, outputs=gr.File())

    # Simulate loading bar
    threading.Thread(target=lambda: async_progress(progress_bar.set_value)).start()

# Launch app
demo.queue().launch()
