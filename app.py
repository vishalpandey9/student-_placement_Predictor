import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import PyPDF2
import google.generativeai as genai

# ==========================================
# PAGE CONFIGURATION & ADVANCED CSS
# ==========================================
st.set_page_config(page_title="AI Placement & ATS System", layout="wide", page_icon="🎓", initial_sidebar_state="expanded")

# Injecting Advanced Custom CSS
st.markdown("""
<style>
    /* 1. Dynamic Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #f0f2f5, #e0e5ec, #ffffff, #f4f7f6);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Dark mode override for the animated background */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: linear-gradient(-45deg, #1a1a2e, #16213e, #0f3460, #1a1a2e);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
    }

    /* 2. Page Transition Animation */
    @keyframes fadeSlideUp {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .main .block-container {
        animation: fadeSlideUp 0.7s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    /* 3. "Hard Hit" Button Animations */
    .stButton>button {
        transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 12px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .stButton>button:active {
        /* The 'hard hit' effect */
        transform: scale(0.92) translateY(4px);
        box-shadow: inset 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Specifically styling the Primary Submit buttons to stand out */
    button[kind="primary"] {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
        border: none;
    }

    /* 4. Enhanced Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0,0,0,0.1);
    }
    
    @media (prefers-color-scheme: dark) {
        section[data-testid="stSidebar"] {
            background-color: rgba(20, 20, 20, 0.9) !important;
            border-right: 1px solid rgba(255,255,255,0.1);
        }
    }

    /* 5. Metric Cards & Diagnostic Boxes */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        padding: 5% 5% 5% 10%;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
    }
    
    /* 6. Polished Header Images */
    img {
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin-bottom: 25px;
        transition: transform 0.5s ease;
    }
    img:hover {
        transform: scale(1.01);
    }
    
    /* 7. Loading Spinner Customization (Pulse Effect) */
    .stSpinner > div > div {
        border-color: #6e8efb transparent #a777e3 transparent !important;
        animation: customSpin 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# ML MODEL SETUP
# ==========================================
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
    
    score = (data['CGPA']*10 + data['DSA_Score'] + data['Internships']*20 + data['Projects']*10)
    threshold = score.median()
    data['Placed'] = (score > threshold).astype(int)
    
    X = data.drop('Placed', axis=1)
    y = data['Placed']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()


# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def analyze_resume_with_gemini(api_key, resume_text, job_desc):
    genai.configure(api_key=api_key)
    try:
        llm = genai.GenerativeModel('gemini-3.6-flash')
        prompt = f"""
        You are an expert IT Recruiter ATS system.
        Analyze this candidate's resume against the following job description.
        
        Job Description: {job_desc}
        Resume: {resume_text}
        
        Provide your output exactly in these 3 sections using Markdown formatting:
        ### 1. Match Percentage
        ### 2. Missing Keywords
        ### 3. Actionable Feedback
        """
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"API Error: {str(e)}"


# ==========================================
# PAGE 1: PREDICTION ANALYTICS
# ==========================================
def page_prediction():
    st.image("https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?auto=format&fit=crop&w=1200&q=80", use_container_width=True)
    st.title("📊 Placement Probability Engine")
    st.markdown("Analyze your academic metrics to receive a precise probability score and a detailed diagnostic report.")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Candidate Metrics")
        with st.form("prediction_form"):
            cgpa = st.slider("Current CGPA", 0.0, 10.0, 7.5, 0.1)
            dsa = st.slider("DSA/Coding Score (out of 100)", 0, 100, 60)
            internships = st.number_input("Number of Internships", 0, 5, 1)
            projects = st.number_input("Number of Major Projects", 0, 10, 2)
            predict_btn = st.form_submit_button("Generate Diagnostic Report", use_container_width=True, type="primary")
            
    with col2:
        st.subheader("AI Prediction Result")
        if predict_btn:
            features = np.array([[cgpa, dsa, internships, projects]])
            prob = model.predict_proba(features)[0][1] * 100
            
            st.metric("Placement Probability", f"{prob:.1f}%")
            
            if prob >= 75:
                st.success("🎉 Excellent! Your profile strongly aligns with tier-1 technical requirements.")
            elif prob >= 50:
                st.warning("⚠️ Average Profile. You will clear resume shortlisting, but technical rounds will be challenging.")
            else:
                st.error("🚨 High Risk. Immediate strategic intervention is required.")

    if predict_btn:
        st.divider()
        st.markdown("### 📈 Detailed Diagnostic & Strategy Report")
        diag_col1, diag_col2 = st.columns(2)
        
        with diag_col1:
            st.markdown("#### 🎯 Where to Focus")
            if dsa < 65:
                st.error("**Critical Bottleneck: DSA Score**\nStart solving 2-3 LeetCode problems daily, focusing heavily on Binary Trees and Graph Algorithms.")
            else:
                st.success("**Strength: DSA Score**\nYour algorithmic logic is solid.")
                
            if projects < 2:
                st.warning("**Area of Improvement: Practical Experience**\nYou need more complex projects. Build full-stack applications or interactive data dashboards.")
            else:
                st.success("**Strength: Project Portfolio**\nYou have a good amount of project work.")

        with diag_col2:
            st.markdown("#### 🛤️ Alternative Pathways")
            if internships >= 1 and projects >= 2 and dsa < 60:
                st.info("**Data Analytics & Web Dev**\nMaster SQL, Python, and robust backends. Roles here prioritize project execution over intense competitive programming.")
            elif dsa > 75 and projects == 0:
                st.info("**Backend Optimization**\nFocus on systems where complex algorithmic logic is highly valued, like quantitative trading tech firms.")
            else:
                st.info("**Technical Consulting**\nCompanies value a balanced profile. Focus on Aptitude, English proficiency, and foundational CS concepts.")


# ==========================================
# PAGE 2: ATS SCANNER
# ==========================================
def page_ats():
    st.image("https://images.unsplash.com/photo-1586281380349-632531db7ed4?auto=format&fit=crop&w=1200&q=80", use_container_width=True)
    st.title("📄 AI Resume ATS Scanner")
    st.markdown("Upload your PDF resume to evaluate it against a specific job description.")
    
    gemini_key = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            gemini_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass 
        
    if not gemini_key:
        st.warning("⚠️ API Key not found! Please configure GEMINI_API_KEY in your Streamlit Cloud Secrets.")
    
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        job_description = st.text_area("Paste Target Job Description", "Software Engineer with Python, SQL, and Data Structures expertise.", height=200)
    with col2:
        uploaded_pdf = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    if st.button("Initialize Deep ATS Scan", type="primary", use_container_width=True):
        if not gemini_key:
            st.error("Cannot run scan: Gemini API Key is missing.")
        elif not uploaded_pdf:
            st.error("Please upload a PDF resume.")
        else:
            with st.spinner("🚀 Extracting text, analyzing semantic matches, and querying LLM..."):
                resume_text = extract_text_from_pdf(uploaded_pdf)
                feedback = analyze_resume_with_gemini(gemini_key, resume_text, job_description)
                
                st.divider()
                st.subheader("🧠 ATS Evaluation Report")
                st.markdown(feedback)


# ==========================================
# PAGE 3: PREPARATION ROADMAP
# ==========================================
def page_roadmap():
    st.image("https://images.unsplash.com/photo-1506784365847-bbad939e9335?auto=format&fit=crop&w=1200&q=80", use_container_width=True)
    st.title("🗺️ Comprehensive Career Roadmap")
    st.markdown("A deep-dive, 6-month curriculum engineered for your target role.")
    
    target_tier = st.selectbox(
        "Select Target Placement Tier:",
        ["Product-Based Companies (SDE)", "Data Analytics & AI Roles", "Service-Based Companies"]
    )
    
    tab1, tab2, tab3 = st.tabs(["Phase 1 (Months 1-2)", "Phase 2 (Months 3-4)", "Phase 3 (Months 5-6)"])
    
    if target_tier == "Product-Based Companies (SDE)":
        with tab1:
            st.write("- **Language Mastery:** Master OOPs (Inheritance, Polymorphism).")
            st.write("- **Basic DSA:** Arrays, Strings, Pointers, Recursion.")
        with tab2:
            st.write("- **Heavy DSA:** Deep dive into Binary Trees, Graphs, DP.")
            st.write("- **Backend Frameworks:** Build scalable APIs (ASP.NET MVC or Node.js).")
        with tab3:
            st.write("- **Project Deployment:** Push 2 major projects to GitHub.")
            st.write("- **CS Fundamentals:** Revise OS, Computer Networks, and DBMS.")

    elif target_tier == "Data Analytics & AI Roles":
        with tab1:
            st.write("- **Python Fundamentals:** Deep dive into Python list comprehensions.")
            st.write("- **Data Libraries:** Fluency in NumPy and Pandas.")
        with tab2:
            st.write("- **Visuals:** Matplotlib and Plotly for charts.")
            st.write("- **ML Algorithms:** Regression, Decision Trees, Time-Series forecasting.")
        with tab3:
            st.write("- **End-to-End Project:** Build a multi-page dashboard application.")
            st.write("- **Business Analytics:** Practice guesstimates and case studies.")

    elif target_tier == "Service-Based Companies":
        with tab1:
            st.write("- **Quantitative Aptitude:** Practice percentages, speed-distance.")
            st.write("- **Logical Reasoning:** Syllogisms, pattern recognition.")
        with tab2:
            st.write("- **Subject Revision:** Operating Systems, Digital Electronics.")
            st.write("- **Web Basics:** DOM, HTML, CSS, basic JavaScript.")
        with tab3:
            st.write("- **Communication:** Practice spoken English and articulation.")
            st.write("- **Resume Polish:** Ensure zero grammatical errors.")


# ==========================================
# APP NAVIGATION ROUTING & FLOATING CHATBOT
# ==========================================
st.sidebar.title("Navigation Menu")
st.sidebar.markdown("**Developer:** Vishal Pandey")
st.sidebar.markdown("**Roll No:** 2400900100155")
st.sidebar.divider()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

with st.sidebar.popover("💬 Open AI Assistant", use_container_width=True):
    st.markdown("### 🤖 Placement Mentor")
    st.caption("Ask me anything about DSA, interviews, or your resume!")
    
    faq_col1, faq_col2 = st.columns(2)
    if faq_col1.button("Improve DSA?", use_container_width=True):
        st.session_state.chat_messages.append({"role": "user", "content": "How can I improve my DSA scores for product-based companies?"})
        st.rerun()
    if faq_col2.button("Best projects?", use_container_width=True):
        st.session_state.chat_messages.append({"role": "user", "content": "What are the best software projects to put on a fresher resume?"})
        st.rerun()
    
    st.divider()
    
    chat_container = st.container(height=350)
    with chat_container:
        for msg in st.session_state.chat_messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your question here...")
        send_btn = st.form_submit_button("Send to AI", use_container_width=True)
        
    if (send_btn and user_input) or (len(st.session_state.chat_messages) > 0 and st.session_state.chat_messages[-1]["role"] == "user" and send_btn == False):
        if send_btn and user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            st.rerun()
            
        try:
            gemini_key = st.secrets.get("GEMINI_API_KEY", "")
            if not gemini_key:
                with chat_container:
                    st.error("API Key missing in Secrets.")
            else:
                with chat_container:
                    with st.spinner("Mentor is typing..."):
                        genai.configure(api_key=gemini_key)
                        llm_chat = genai.GenerativeModel('gemini-3.6-flash')
                        
                        last_user_msg = st.session_state.chat_messages[-1]["content"]
                        system_prompt = f"You are a helpful college placement mentor for a B.Tech CSE student. Answer concisely. Student asks: {last_user_msg}"
                        
                        response = llm_chat.generate_content(system_prompt)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response.text})
                        st.rerun()
                        
        except Exception as e:
            with chat_container:
                st.error(f"Chat Error: {str(e)}")

st.sidebar.divider()

pages = {
    "Core Applications": [
        st.Page(page_prediction, title="Placement Predictor", icon="📊"),
        st.Page(page_ats, title="LLM ATS Scanner", icon="📄"),
    ],
    "Career Guidance": [
        st.Page(page_roadmap, title="Prep Roadmap", icon="🗺️")
    ]
}

pg = st.navigation(pages)
pg.run()
