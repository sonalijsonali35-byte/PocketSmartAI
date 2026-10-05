# Pocket Smart AI
Flask + SQLite + any OpenAI-compatible LLM.

    python -m venv venv
    venv\Scripts\activate        # macOS/Linux: source venv/bin/activate
    pip install -r requirements.txt
    # edit .env (SECRET_KEY, LLM_API_KEY)
    python app.py                # http://127.0.0.1:5000
