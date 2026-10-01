# 🔎 AI Research Agent

Single-agent research app built with **CrewAI**, **Groq (openai/gpt-oss-120b)**, **DuckDuckGo search** and **Streamlit**.

## Run locally
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then add your Groq key
streamlit run app.py
```

## Deploy
Push to GitHub -> share.streamlit.io -> New app -> choose repo, branch `main`, file `app.py`.
In *Advanced settings* choose Python 3.12 and paste `GROQ_API_KEY = "..."` into Secrets.
