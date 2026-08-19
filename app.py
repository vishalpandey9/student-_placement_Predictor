import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import PyPDF2
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
# This must be the very first Streamlit command
st.set_page_config(page_title="AI Placement & ATS System", layout="wide", page_icon="🎓")

# --- ML MODEL SETUP (Cached so it only trains once) ---
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 500
    
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
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# --- UTILITY FUNCTIONS ---
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def analyze_resume_with_gemini(api_key, resume_text, job_desc):
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""
    You are an expert IT Recruiter ATS system.
    Analyze this candidate's resume against the following job description.
    
    Job Description: {job_desc}
    Resume: {resume_text}
    
    Provide your output exactly in these 3 sections using Markdown formatting:
    ### 1. Match Percentage
    (Provide a single percentage score based on skills alignment)
    ### 2. Missing Keywords
    (List critical skills/tools required in the JD but missing from the resume)
    ### 3. Actionable Feedback
    (Give 2-3 brief tips to improve the resume format or content)
    """
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"API Error: {str(e)}"

# ==========================================
# PAGE 1: PREDICTION ANALYTICS
# ==========================================
def page_prediction():
    st.title("📊 Student Placement Analytics")
    st.markdown("Enter your academic and technical metrics to predict your placement probability using a trained Random Forest model.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Metrics")
        with st.form("prediction_form"):
            cgpa = st.slider("Current CGPA", 0.0, 10.0, 7.5, 0.1)
            dsa = st.slider("DSA/Coding Score (out of 100)", 0, 100, 60)
            internships = st.number_input("Number of Internships", 0, 5, 1)
            projects = st.number_input("Number of Major Projects", 0, 10, 2)
            predict_btn = st.form_submit_button("Predict Probability")
            
    with col2:
        st.subheader("Prediction Results")
        if predict_btn:
            features = np.array([[cgpa, dsa, internships, projects]])
            prob = model.predict_proba(features)[0][1] * 100
            
            st.metric("Placement Probability", f"{prob:.1f}%")
            
            if prob >= 70:
                st.success("🎉 Strong Profile! You are highly likely to clear technical cut-offs.")
                st.balloons()
            elif prob >= 50:
                st.warning("⚠️ Average Profile. Focus on increasing your DSA score or adding a project.")
            else:
                st.error("🚨 Needs Improvement. High risk of screening rejection. See the Roadmap tab for a prep strategy.")

# ==========================================
# PAGE 2: ATS SCANNER
# ==========================================
def page_ats():
    st.title("📄 AI Resume ATS Scanner")
    st.markdown("Upload your PDF resume to evaluate it against a specific job description using Google Gemini.")
    
    # Safely initialize the key first to prevent NameErrors
    gemini_key = None
    try:
        # Check if secrets exist and grab the key
        if "GEMINI_API_KEY" in st.secrets:
            gemini_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass # If there's any issue reading secrets, gemini_key remains None
        
    # Show a warning if the key is missing so you know why it won't run
    if not gemini_key:
        st.warning("⚠️ API Key not found! Please configure GEMINI_API_KEY in your Streamlit Cloud Secrets.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        job_description = st.text_area("Paste Target Job Description", "Software Engineer with Python, SQL, and Data Structures expertise.", height=200)
    with col2:
        uploaded_pdf = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    if st.button("Run ATS Scan", type="primary"):
        if not gemini_key:
            st.error("Cannot run scan: Gemini API Key is missing. Check your Streamlit Cloud Settings.")
        elif not uploaded_pdf:
            st.error("Please upload a PDF resume.")
        else:
            with st.spinner("Parsing PDF and querying Gemini LLM..."):
                resume_text = extract_text_from_pdf(uploaded_pdf)
                feedback = analyze_resume_with_gemini(gemini_key, resume_text, job_description)
                
                st.divider()
                st.subheader("🧠 ATS Evaluation Report")
                st.markdown(feedback)
# ==========================================
# PAGE 3: PREPARATION ROADMAP
# ==========================================
def page_roadmap():
    st.title("🗺️ Personalized Prep Roadmap")
    st.markdown("Select your target role to generate a specific timeline for placement preparation.")
    
    target_tier = st.selectbox(
        "Select Target Placement Tier:",
        ["Product-Based Companies (SDE)", "Data Analytics / AI Roles", "Service-Based Companies (TCS, Wipro)"]
    )
    
    st.divider()
    
    if target_tier == "Product-Based Companies (SDE)":
        st.subheader("Roadmap: Product-Based SDE")
        st.markdown("""
        *   **Month 1-2:** Master Data Structures (Arrays, Strings, Linked Lists) and Object-Oriented Programming (Java/C++).
        *   **Month 3-4:** Advanced DSA (Trees, Graphs, DP). Solve Top 150 LeetCode problems.
        *   **Month 5:** Build 2 full-stack projects (e.g., ASP.NET MVC or Python/React).
        *   **Month 6:** System Design Basics (Load balancers, DB indexing) and mock technical interviews.
        """)
        
    elif target_tier == "Data Analytics / AI Roles":
        st.subheader("Roadmap: Data & Analytics")
        st.markdown("""
        *   **Month 1-2:** Python mastery (NumPy, Pandas, Plotly) and advanced SQL (Window functions, Joins).
        *   **Month 3-4:** Machine Learning algorithms (Regression, Random Forest) and Time-Series forecasting.
        *   **Month 5:** Build end-to-end interactive dashboards using Streamlit (like a Stock Predictor).
        *   **Month 6:** Build a portfolio website and practice Guesstimates and SQL case studies.
        """)
        
    elif target_tier == "Service-Based Companies (TCS, Wipro)":
        st.subheader("Roadmap: Service-Based IT")
        st.markdown("""
        *   **Month 1-2:** Quantitative Aptitude and Logical Reasoning practice.
        *   **Month 3:** Basic programming logic (Loops, Arrays, Strings in C, C++, or Java).
        *   **Month 4:** Core CS Subjects (OS, DBMS, Computer Networks).
        *   **Month 5-6:** Spoken English practice, resume building, and HR mock interviews.
        """)

# ==========================================
# APP NAVIGATION ROUTING
# ==========================================
# Sidebar branding
st.sidebar.title("System Navigation")
st.sidebar.markdown("**Developer:** Vishal Pandey")
st.sidebar.markdown("**Roll Number:** 2400900100155")
st.sidebar.divider()

# Define the pages and group them
pages = {
    "Core Tools": [
        st.Page(page_prediction, title="Placement Predictor", icon="📊"),
        st.Page(page_ats, title=" ATS Scanner", icon="📄"),
    ],
    "Guidance": [
        st.Page(page_roadmap, title="Prep Roadmap", icon="🗺️")
    ]
}

# Run the navigation router
pg = st.navigation(pages)
pg.run()
