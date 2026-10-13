# Local RAG Demo

A tiny Retrieval-Augmented Generation app: Streamlit + Ollama + FAISS. 100% local, no API keys.

## Setup

1. Install [Ollama](https://ollama.com), then pull the models:
   ```powershell
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```
2. Create and activate a virtual environment (Windows PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   If activation is blocked: `Set-ExecutionPolicy -Scope Process RemoteSigned`
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run:
   ```powershell
   streamlit run app.py
   ```

## Troubleshooting

| Problem | Fix |
|---|---|
| Connection error / "Cannot reach Ollama" | Start Ollama: `ollama serve` (or open the Ollama app) |
| "Model not found" | Run `ollama pull <model>` (e.g. `llama3.2`, `nomic-embed-text`) |
| Slow on CPU | Use a smaller chunk size/top-k, a smaller document, or a smaller chat model |
