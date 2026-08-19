import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import PyPDF2
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Placement & ATS System", layout="wide", page_icon="🎓")

# College Project Branding
st.sidebar.title("Project Details")
st.sidebar.markdown("**Developer:** Vishal Pandey")
st.sidebar.markdown("**Roll Number:** 2400900100155")
st.sidebar.markdown("---")
st.sidebar.info("This system uses a Random Forest Classifier for placement prediction and Google Gemini for ATS resume parsing.")

# --- 1. ML MODEL SETUP (Auto-trained on synthetic data for instant use) ---
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 500
    
    # Simulating student data metrics
    data = pd.DataFrame({
        'CGPA': np.random.uniform(5.0, 10.0, n_samples),
        'DSA_Score': np.random.randint(10, 100, n_samples),
        'Internships': np.random.randint(0, 4, n_samples),
        'Projects': np.random.randint(1, 6, n_samples),
    })
    
    # Logic defining placement probability based on metrics
    score = (data['CGPA']*10 + data['DSA_Score'] + data['Internships']*20 + data['Projects']*10)
    threshold = score.median()
    data['Placed'] = (score > threshold).astype(int)
    
    X = data.drop('Placed', axis=1)
    y = data['Placed']
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# --- 2. LLM ATS UTILITIES ---
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def analyze_resume_with_gemini(api_key, resume_text, job_desc):
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are an expert Applicant Tracking System (ATS).
    Analyze this candidate's resume against the following job description.
    
    Job Description:
    {job_desc}
    
    Resume:
    {resume_text}
    
    Provide your output in exactly 3 sections:
    1. Match Percentage (e.g., 75%)
    2. Missing Keywords (Skills required in JD but missing from resume)
    3. Actionable Formatting & Content Feedback
    """
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"API Error: {str(e)}"

# --- 3. FRONTEND DASHBOARD ---
st.title("🎓 AI Student Placement Predictor & LLM ATS")
st.markdown("A dual-pipeline architecture analyzing both structured academic data and unstructured resume data.")

col1, col2 = st.columns(2)

# Left Column: ML Prediction
with col1:
    st.header("📊 1. Placement Predictor")
    st.markdown("Enter academic and technical metrics to predict placement probability.")
    
    with st.form("prediction_form"):
        cgpa = st.slider("Current CGPA", 0.0, 10.0, 7.5, 0.1)
        dsa = st.slider("DSA/Coding Score (out of 100)", 0, 100, 60)
        internships = st.number_input("Number of Internships", 0, 5, 1)
        projects = st.number_input("Number of Major Projects", 0, 10, 2)
        predict_btn = st.form_submit_button("Predict Probability")
        
        if predict_btn:
            features = np.array([[cgpa, dsa, internships, projects]])
            prob = model.predict_proba(features)[0][1] * 100
            
            if prob >= 55:
                st.success(f"### 🎉 High Probability: {prob:.1f}% chance of placement.")
                st.balloons()
            else:
                st.warning(f"### ⚠️ Needs Improvement: {prob:.1f}% chance of placement.")
                st.info("Tip: Boost your DSA score and complete more internships to improve odds.")

# Right Column: LLM ATS
with col2:
    st.header("📄 2. LLM Resume ATS")
    st.markdown("Upload a PDF resume to evaluate it against a specific job description.")
    
    gemini_key = st.text_input("Enter Google Gemini API Key", type="password")
    job_description = st.text_area("Paste Job Description", "Software Engineer with Python, Machine Learning, and Data Structures expertise.")
    uploaded_pdf = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    if st.button("Analyze Resume with AI"):
        if not gemini_key:
            st.error("Please provide a Gemini API Key.")
        elif not uploaded_pdf:
            st.error("Please upload a PDF resume.")
        else:
            with st.spinner("Parsing PDF and querying Gemini LLM..."):
                resume_text = extract_text_from_pdf(uploaded_pdf)
                feedback = analyze_resume_with_gemini(gemini_key, resume_text, job_description)
                st.write("### 🧠 ATS Analysis Results")
                st.write(feedback)
