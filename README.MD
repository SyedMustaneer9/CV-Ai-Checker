# AI Resume ATS Checker

Upload a resume (PDF, DOCX or TXT) and get:

- An estimated **ATS score** (0-100) with a 5-part breakdown
- Strengths, weaknesses and **missing keywords**
- Prioritised **improvements** and before/after **bullet rewrites**
- Quick rule-based checks (contact info, sections, length, bullets, metrics)
- A downloadable Markdown report

Optionally paste a job description to get keyword matching against that role.

Built with [Streamlit](https://streamlit.io) and the Google Gemini API (Flash model).

> The score is an AI estimate. No real ATS vendor's algorithm is used, so treat it as guidance, not a guarantee.

## Run locally

1. Get a free API key at https://aistudio.google.com/apikey
2. Install and run:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

3. Paste your key in the sidebar, or set it beforehand:

```bash
export GEMINI_API_KEY="your-key"        # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
```

Or create `.streamlit/secrets.toml` (never commit this file):

```toml
GEMINI_API_KEY = "your-key"
```

## Configuration

| Setting | How | Default |
|---|---|---|
| API key | Sidebar, `GEMINI_API_KEY` env var, or Streamlit secrets | none |
| Model | Sidebar field, or `GEMINI_MODEL` env var | `gemini-3.5-flash` |

If your key or region does not have access to the default model, change it in the sidebar (for example `gemini-3.6-flash`).

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (repo must contain `app.py` and `requirements.txt`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings > Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Project files

- `app.py` - the Streamlit app
- `requirements.txt` - Python dependencies
- `README.md` - this file

## Limits and privacy

- Max upload size: 5 MB. Scanned or image-only resumes cannot be read (an ATS cannot read them either).
- Resume text is sent to Google's Gemini API for analysis. Do not upload anything you are not comfortable sharing with that service.
- Nothing is stored by the app; results live only in your browser session.
