import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from docx import Document

st.set_page_config(page_title="📄 ATS Resume Checker", page_icon="📊")

# --- API Key Setup ---
api_key = st.secrets.get("GOOGLE_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("🔑 Enter your Google Gemini API Key", type="password")

if not api_key:
    st.sidebar.warning("⚠️ Add your API key to use the app")
else:
    genai.configure(api_key=api_key)
    # ✅ Using Gemini 3.5 Flash as requested
    model = genai.GenerativeModel("gemini-3.5-flash")

# --- Text Extraction Functions ---
def extract_pdf_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_docx_text(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# --- AI Analysis Function ---
def analyze_resume(resume_text):
    prompt = f"""
You are an expert ATS (Applicant Tracking System) analyzer and career coach.

Analyze the resume below and return your evaluation in THIS EXACT format:

---
📊 ATS SCORE: XX/100

📈 SCORE BREAKDOWN:
- Keyword Optimization: XX/30
- Formatting & Structure: XX/25
- Content & Impact: XX/25
- Clarity & Conciseness: XX/20

✅ STRENGTHS:
- [bullet point]

🔧 AREAS TO IMPROVE:
- [bullet point]

💡 SPECIFIC SUGGESTIONS:
- [concrete, actionable tip]

⚠️ COMMON ATS ISSUES:
- [list any formatting risks]

---
Resume content:
{resume_text}
---

Be honest, specific, and constructive. Give numbers and clear explanations.
"""
    response = model.generate_content(prompt)
    return response.text

# --- Main UI ---
st.title("📄 ATS Resume Score Checker & AI Advisor")
st.markdown("Upload your resume (PDF or DOCX) to get an estimated ATS score and personalized improvement tips.")

uploaded_file = st.file_uploader("Upload your Resume", type=["pdf", "docx"])

if uploaded_file and api_key:
    with st.spinner("🔍 Reading resume and analyzing..."):
        try:
            # Extract text
            if uploaded_file.name.endswith(".pdf"):
                resume_text = extract_pdf_text(uploaded_file)
            else:
                resume_text = extract_docx_text(uploaded_file)

            # Validate extracted text
            if len(resume_text.strip()) < 50:
                st.error("❌ Could not extract enough text. Make sure this is not a scanned/image-only file.")
            else:
                # Get AI analysis
                result = analyze_resume(resume_text)
                st.success("✅ Analysis Complete!")
                st.markdown(result)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

elif not api_key:
    st.info("AQ.Ab8RN6IbuzcSbBloSOHQJ0PxutTICtPlN0LAr75akgQa4qHpIg")

# --- Footer ---
st.markdown("---")
st.caption("⚠️ This is an AI-powered estimation tool — not an official ATS system. Results are for guidance only.")
