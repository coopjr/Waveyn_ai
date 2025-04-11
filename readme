# 🚀 Multi-Model AI Chatbot with Voice & History (Hugging Face + Local Docker)

This project is a modular, multi-model AI chatbot built with Gradio. It supports:

- ✅ Multiple quantized models (Mistral 7B, LLaMA 7B, Qwen 1.8B multilingual)
- ✅ Voice input (microphone)
- ✅ Live token streaming with typing indicator
- ✅ User session separation
- ✅ Downloadable chat history (.txt file)
- ✅ Fancy UI with dark/light mode toggle, progress bar, and model info card

---

## 🌐 Deployment Options

### Option 1: Run Locally with Docker

1. **Clone the repository**

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

2. **Build the Docker image**

```bash
docker build -t ai-chatbot .
```

3. **Run the container**

```bash
docker run -p 7860:7860 ai-chatbot
```

4. **Open your browser**

Go to: [http://localhost:7860](http://localhost:7860)

### Option 2: Deploy to Hugging Face Space

- Just upload your files to your Hugging Face Space.
- Make sure you select **Docker** as the runtime.
- Hugging Face will auto-detect the `Dockerfile` and launch the app!

---

## ⚙️ Project Structure

```
├── app.py               # Main application code
├── Dockerfile           # Containerization setup
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── chat_history_*.txt   # User chat history (generated dynamically)
```

---

## 💡 Features

- 🔄 **Model Switching**: Seamlessly switch between models from the dropdown menu.
- 🧠 **Session Management**: Keeps your chats separate per user.
- 🎙️ **Voice Input**: Speak to the chatbot directly.
- 📄 **Download Chat History**: Export your conversations as .txt files.
- 🎨 **Clean UI**: Light/Dark mode, progress bar, and model info card.

---

## 🧩 Add or Remove Models

Update the `MODELS` dictionary in `app.py`:

```python
MODELS = {
    "Mistral 7B Q4": "mistral-model-path",
    "LLaMA 7B Q4": "llama-model-path",
    "Qwen 1.8B Q4 (Multilingual)": "qwen-model-path"
}
```

---

## 📄 License

MIT License. Free to use and modify!

---

## 🙌 Contributing

PRs are welcome! Let’s make this chatbot even better 🚀

---

## ✨ Enjoy chatting!

> Built with ❤️ for fast, friendly, and flexible AI conversations.
