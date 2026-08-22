import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import PyPDF2
import google.generativeai as genai
from streamlit_option_menu import option_menu

# ==========================================
# PAGE CONFIGURATION & ADVANCED CSS
# ==========================================
st.set_page_config(page_title="AI Placement & ATS System", layout="wide", page_icon="🎓")

# Injecting Advanced Custom CSS for Ultimate Responsiveness & Floating Chat
st.markdown("""
<style>
    /* 1. Dynamic Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #f8f9fa, #e9ecef, #f1f3f5, #ffffff);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: linear-gradient(-45deg, #0f172a, #1e293b, #334155, #0f172a);
        }
    }

    /* 2. Page Transition Animation */
    @keyframes fadeSlideUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .main .block-container {
        animation: fadeSlideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        padding-top: 2rem; 
        padding-bottom: 100px; /* Space for the floating chat */
    }
    
    /* 3. Floating Right-Side Chatbot (The Magic) */
    div[data-testid="stPopover"] {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 99999;
    }
    div[data-testid="stPopover"] > button {
        background: linear-gradient(135deg, #6e8efb, #a777e3) !important;
        color: white !important;
        border-radius: 50px !important;
        padding: 15px 30px !important;
        box-shadow: 0 10px 25px rgba(110, 142, 251, 0.4) !important;
        font-size: 16px !important;
        font-weight: bold !important;
        border: none !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div[data-testid="stPopover"] > button:hover {
        transform: scale(1.05) translateY(-4px) !important;
        box-shadow: 0 15px 35px rgba(110, 142, 251, 0.6) !important;
    }

    /* 4. Button Interactions */
    .stButton>button {
        transition: all 0.2s ease;
        border-radius: 10px;
        font-weight: 600;
    }
    .stButton>button:active {
        transform: scale(0.95);
    }
    
    /* 5. Glassmorphism Metric Cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.05);
    }

    /* 6. Polished Images */
    img {
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        object-fit: cover;
        width: 100%;
        height: 300px; /* Fixed height for clean aesthetic */
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# ML MODEL SETUP (Cached)
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
# VISIBLE TOP NAVIGATION BAR
# ==========================================
# College Project Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### 🎓 AI-Powered Placement & ATS Platform")
with col2:
    st.markdown("<div style='text-align: right; color: gray;'><strong>Developer:</strong> Vishal Pandey<br><strong>Roll No:</strong> 2400900100155</div>", unsafe_allow_html=True)

st.write("") # Spacer

selected_page = option_menu(
    menu_title=None,
    options=["Predictor Engine", "Resume ATS", "Career Roadmap"],
    icons=["bar-chart-line", "file-earmark-person", "map"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": "#6e8efb", "font-size": "18px"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#e9ecef"},
        "nav-link-selected": {"background-color": "#6e8efb", "color": "white"},
    }
)

st.divider()

# ==========================================
# PAGE CONTROLLERS
# ==========================================

if selected_page == "Predictor Engine":
    # Premium Data Image
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1600&q=80", use_container_width=True)
    st.title("📊 Placement Probability Engine")
    st.markdown("Analyze your academic metrics to receive a precise probability score and a detailed diagnostic report.")
    
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        with st.form("prediction_form"):
            cgpa = st.slider("Current CGPA", 0.0, 10.0, 7.5, 0.1)
            dsa = st.slider("DSA/Coding Score (out of 100)", 0, 100, 60)
            internships = st.number_input("Number of Internships", 0, 5, 1)
            projects = st.number_input("Number of Major Projects", 0, 10, 2)
            predict_btn = st.form_submit_button("Generate Diagnostic Report", use_container_width=True, type="primary")
            
    with col2:
        if predict_btn:
            features = np.array([[cgpa, dsa, internships, projects]])
            prob = model.predict_proba(features)[0][1] * 100
            
            st.metric("Probability Score", f"{prob:.1f}%")
            
            if prob >= 75:
                st.success("🎉 Excellent! Your profile strongly aligns with tier-1 technical requirements.")
            elif prob >= 50:
                st.warning("⚠️ Average Profile. Technical rounds will require intensive preparation.")
            else:
                st.error("🚨 High Risk. Immediate strategic intervention is required.")

    if predict_btn:
        st.divider()
        st.markdown("### 📈 Strategy Report")
        diag_col1, diag_col2 = st.columns(2)
        
        with diag_col1:
            st.markdown("#### 🎯 Execution Focus")
            if dsa < 65:
                st.error("**Critical: DSA Score**\nSolve 2-3 algorithmic problems daily, focusing on Trees and Graphs.")
            else:
                st.success("**Strength: DSA Score**\nYour algorithmic logic is highly competitive.")
                
            if projects < 2:
                st.warning("**Improvement: Practical Work**\nBuild complex, deployed full-stack applications.")
            else:
                st.success("**Strength: Portfolio**\nYour project volume is excellent.")

        with diag_col2:
            st.markdown("#### 🛤️ Alternative Pathways")
            if internships >= 1 and projects >= 2 and dsa < 60:
                st.info("**Data Analytics & Web Dev**\nMaster SQL, Python, and robust backends.")
            else:
                st.info("**Technical Consulting**\nFocus on Aptitude, Communication, and foundational CS concepts.")


elif selected_page == "Resume ATS":
    # Premium Corporate Desk Image
    st.image("https://images.unsplash.com/photo-1586281380117-5a60ae2050cc?auto=format&fit=crop&w=1600&q=80", use_container_width=True)
    st.title("📄 AI Resume ATS Scanner")
    st.markdown("Evaluate your resume against specific job descriptions to bypass automated filters.")
    
    gemini_key = st.secrets.get("GEMINI_API_KEY", None)
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


elif selected_page == "Career Roadmap":
    # Premium Architecture/Path Image
    st.image("https://images.unsplash.com/photo-1507679799987-c73779587ccf?auto=format&fit=crop&w=1600&q=80", use_container_width=True)
    st.title("🗺️ Comprehensive Career Roadmap")
    st.markdown("A deep-dive, 6-month execution plan engineered for your target industry.")
    
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
# THE FLOATING RIGHT-SIDE AI MENTOR
# ==========================================
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# This popover is completely controlled by the CSS at the top of the file
with st.popover("💬 AI Assistant"):
    st.markdown("### 🤖 Placement Mentor")
    st.caption("Ask me anything about your career journey!")
    
    faq_col1, faq_col2 = st.columns(2)
    if faq_col1.button("Improve DSA?", use_container_width=True, key="faq1"):
        st.session_state.chat_messages.append({"role": "user", "content": "How can I improve my DSA scores?"})
        st.rerun()
    if faq_col2.button("Best projects?", use_container_width=True, key="faq2"):
        st.session_state.chat_messages.append({"role": "user", "content": "What are the best software projects?"})
        st.rerun()
    
    st.divider()
    
    chat_container = st.container(height=300)
    with chat_container:
        for msg in st.session_state.chat_messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your question here...")
        send_btn = st.form_submit_button("Send", use_container_width=True)
        
    if (send_btn and user_input) or (len(st.session_state.chat_messages) > 0 and st.session_state.chat_messages[-1]["role"] == "user" and send_btn == False):
        if send_btn and user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            st.rerun()
            
        try:
            gemini_key = st.secrets.get("GEMINI_API_KEY", "")
            if not gemini_key:
                with chat_container:
                    st.error("API Key missing.")
            else:
                with chat_container:
                    with st.spinner("Typing..."):
                        genai.configure(api_key=gemini_key)
                        llm_chat = genai.GenerativeModel('gemini-3.6-flash')
                        last_user_msg = st.session_state.chat_messages[-1]["content"]
                        system_prompt = f"You are a helpful college placement mentor. Answer concisely. Student asks: {last_user_msg}"
                        response = llm_chat.generate_content(system_prompt)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response.text})
                        st.rerun()
                        
        except Exception as e:
            with chat_container:
                st.error("Connection Error.")
