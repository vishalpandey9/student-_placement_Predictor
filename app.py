import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import PyPDF2
import google.generativeai as genai

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS ANIMATIONS
# ==========================================
st.set_page_config(page_title="AI Placement & ATS System", layout="wide", page_icon="🎓")

# Injecting Custom CSS for Animations and Responsiveness
st.markdown("""
<style>
    /* Fade-in and slide-up animation for page transitions */
    @keyframes fadeSlideUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Apply animation to the main content container */
    .stApp > header { background-color: transparent; }
    .main .block-container {
        animation: fadeSlideUp 0.6s ease-out;
    }
    
    /* Hover animations for all buttons */
    .stButton>button {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Styling for metric cards to make them pop */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    
    /* Dark mode support for metric cards */
    @media (prefers-color-scheme: dark) {
        div[data-testid="metric-container"] {
            background-color: #1e1e1e;
            border-color: #333;
        }
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# ML MODEL SETUP (Cached for performance)
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
        # Strictly using the required model version
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
    st.title("📊 Student Placement Analytics")
    st.markdown("Enter your academic and technical metrics to predict your placement probability.")
    
    # Using responsive gap columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Input Metrics")
        with st.form("prediction_form"):
            cgpa = st.slider("Current CGPA", 0.0, 10.0, 7.5, 0.1)
            dsa = st.slider("DSA/Coding Score (out of 100)", 0, 100, 60)
            internships = st.number_input("Number of Internships", 0, 5, 1)
            projects = st.number_input("Number of Major Projects", 0, 10, 2)
            predict_btn = st.form_submit_button("Predict Probability", use_container_width=True)
            
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
                st.error("🚨 Needs Improvement. High risk of screening rejection. See the Roadmap tab.")


# ==========================================
# PAGE 2: ATS SCANNER
# ==========================================
def page_ats():
    st.title("📄 AI Resume ATS Scanner")
    st.markdown("Upload your PDF resume to evaluate it against a specific job description using Google Gemini.")
    
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
    
    if st.button("Run ATS Scan", type="primary", use_container_width=True):
        if not gemini_key:
            st.error("Cannot run scan: Gemini API Key is missing.")
        elif not uploaded_pdf:
            st.error("Please upload a PDF resume.")
        else:
            with st.spinner("Analyzing resume formatting and parsing keywords..."):
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
        *   **Month 1-2:** Master Data Structures and OOPs concepts.
        *   **Month 3-4:** Advanced DSA (Trees, Graphs, DP). Solve LeetCode medium/hard.
        *   **Month 5:** Build 2 full-stack projects (e.g., ASP.NET MVC or Python/React).
        *   **Month 6:** System Design Basics and mock technical interviews.
        """)
        
    elif target_tier == "Data Analytics / AI Roles":
        st.subheader("Roadmap: Data & Analytics")
        st.markdown("""
        *   **Month 1-2:** Python mastery (NumPy, Pandas, Plotly) and advanced SQL.
        *   **Month 3-4:** Machine Learning algorithms and Time-Series forecasting (ARIMA).
        *   **Month 5:** Build end-to-end interactive dashboards using Streamlit.
        *   **Month 6:** Build a portfolio website and practice Guesstimates.
        """)
        
    elif target_tier == "Service-Based Companies (TCS, Wipro)":
        st.subheader("Roadmap: Service-Based IT")
        st.markdown("""
        *   **Month 1-2:** Quantitative Aptitude and Logical Reasoning practice.
        *   **Month 3:** Basic programming logic (Loops, Arrays, Strings).
        *   **Month 4:** Core CS Subjects (OS, DBMS, Computer Networks, Theory of Automata).
        *   **Month 5-6:** Communication skills, resume formatting, and HR mock interviews.
        """)


# ==========================================
# APP NAVIGATION ROUTING & FLOATING CHATBOT
# ==========================================
st.sidebar.title("System Navigation")
st.sidebar.markdown("**Developer:** Vishal Pandey")
st.sidebar.markdown("**Roll Number:** 2400900100155")
st.sidebar.divider()

# --- FLOATING CHATBOT WIDGET ---
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

with st.sidebar.popover("💬 Open AI Assistant", use_container_width=True):
    st.markdown("### 🤖 Placement Mentor")
    st.caption("Ask me anything about DSA, interviews, or your resume!")
    
    # FAQ Quick Action Buttons
    st.markdown("**Frequently Asked Questions:**")
    faq_col1, faq_col2 = st.columns(2)
    if faq_col1.button("How to improve DSA?"):
        st.session_state.chat_messages.append({"role": "user", "content": "How can I improve my DSA scores for product-based companies?"})
        st.rerun()
    if faq_col2.button("Best projects?"):
        st.session_state.chat_messages.append({"role": "user", "content": "What are the best software projects to put on a fresher resume?"})
        st.rerun()
    
    st.divider()
    
    # Chat History Container
    chat_container = st.container(height=350)
    with chat_container:
        for msg in st.session_state.chat_messages:
            st.chat_message(msg["role"]).write(msg["content"])
            
    # Chat Input Form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your question here...")
        send_btn = st.form_submit_button("Send to AI", use_container_width=True)
        
    # Process AI Response
    if (send_btn and user_input) or (len(st.session_state.chat_messages) > 0 and st.session_state.chat_messages[-1]["role"] == "user" and send_btn == False):
        
        # If triggered by send button, append user input
        if send_btn and user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            st.rerun()
            
        # Get AI Response for the last user message
        try:
            gemini_key = st.secrets.get("GEMINI_API_KEY", "")
            if not gemini_key:
                with chat_container:
                    st.error("API Key missing in Secrets.")
            else:
                with chat_container:
                    with st.spinner("Thinking..."):
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

# --- MULTI-PAGE ROUTING ---
pages = {
    "Core Tools": [
        st.Page(page_prediction, title="Placement Predictor", icon="📊"),
        st.Page(page_ats, title="LLM ATS Scanner", icon="📄"),
    ],
    "Guidance": [
        st.Page(page_roadmap, title="Prep Roadmap", icon="🗺️")
    ]
}

pg = st.navigation(pages)
pg.run()
