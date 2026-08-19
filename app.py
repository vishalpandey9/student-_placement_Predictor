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

st.markdown("""
<style>
    /* Fade-in and slide-up animation for page transitions */
    @keyframes fadeSlideUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .stApp > header { background-color: transparent; }
    .main .block-container {
        animation: fadeSlideUp 0.6s ease-out;
    }
    
    /* Hover animations for buttons */
    .stButton>button {
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Styling for metric cards and images */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    
    img {
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
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
    st.title("📊 Placement Probability & Diagnostic Engine")
    st.markdown("Analyze your academic metrics to receive a precise probability score and a detailed diagnostic report on where to focus your efforts.")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Candidate Metrics")
        with st.form("prediction_form"):
            cgpa = st.slider("Current CGPA", 0.0, 10.0, 7.5, 0.1)
            dsa = st.slider("DSA/Coding Score (out of 100)", 0, 100, 60)
            internships = st.number_input("Number of Internships", 0, 5, 1)
            projects = st.number_input("Number of Major Projects", 0, 10, 2)
            predict_btn = st.form_submit_button("Generate Diagnostic Report", use_container_width=True)
            
    with col2:
        st.subheader("AI Prediction")
        if predict_btn:
            features = np.array([[cgpa, dsa, internships, projects]])
            prob = model.predict_proba(features)[0][1] * 100
            
            st.metric("Placement Probability", f"{prob:.1f}%")
            
            if prob >= 75:
                st.success("🎉 Excellent! Your profile strongly aligns with tier-1 technical requirements.")
            elif prob >= 50:
                st.warning("⚠️ Average Profile. You will clear resume shortlisting, but technical rounds will be challenging.")
            else:
                st.error("🚨 High Risk. Immediate strategic intervention is required to secure off-campus or on-campus roles.")

    # Extended Diagnostic Report below the columns
    if predict_btn:
        st.divider()
        st.markdown("### 📈 Detailed Diagnostic & Strategy Report")
        
        diag_col1, diag_col2 = st.columns(2)
        
        with diag_col1:
            st.markdown("#### 🎯 Where to Focus & How to Improve")
            
            if dsa < 65:
                st.error("**Critical Bottleneck: DSA Score**\nYour coding proficiency is below the standard 65+ threshold. Start solving 2-3 LeetCode problems daily, focusing heavily on Binary Trees, Graph Algorithms, and Dynamic Programming. Implement logic first on paper before coding.")
            else:
                st.success("**Strength: DSA Score**\nYour algorithmic logic is solid. Maintain this by participating in weekly coding contests.")
                
            if projects < 2:
                st.warning("**Area of Improvement: Practical Experience**\nYou need more complex projects. Move beyond basic HTML/CSS. Build full-stack applications using robust frameworks like ASP.NET MVC or create interactive data dashboards using Python, Pandas, and Streamlit.")
            else:
                st.success("**Strength: Project Portfolio**\nYou have a good amount of project work. Ensure you can explain the architecture and database schema in depth during an interview.")
                
            if cgpa < 7.0:
                st.warning("**Area of Improvement: Academic Cut-offs**\nA CGPA below 7.0 may disqualify you from some initial corporate screening rounds. Focus on core subjects (Operating Systems, DBMS, Computer Architecture) in your upcoming semesters to buffer this.")

        with diag_col2:
            st.markdown("#### 🛤️ Alternative Career Pathways")
            st.markdown("If traditional SDE (Software Development Engineer) roles feel too competitive, consider leveraging your unique metric mix into these high-growth areas:")
            
            # Dynamic career logic
            if internships >= 1 and projects >= 2 and dsa < 60:
                st.info("**Data Analytics & Engineering**\nLeverage your project experience toward data. Master SQL, Python data libraries (NumPy, Matplotlib, Plotly), and build visual prediction models. Roles in this field often prioritize data manipulation over intense competitive programming.")
                st.info("**Full-Stack / Web Development**\nFocus entirely on the web ecosystem. Build robust backends with C# and ASP.NET or Node.js. Showcase deployed, working software. Portfolios often override low DSA scores here.")
            elif dsa > 75 and projects == 0:
                st.info("**Backend Optimization / Competitive Programmer**\nFocus purely on backend systems where complex algorithmic logic is highly valued. Alternatively, look at quantitative trading tech firms that heavily test problem-solving over web development.")
            else:
                st.info("**Technical Consulting / Service-Based Architecture**\nCompanies like TCS, Wipro, or Infosys value a balanced profile (steady CGPA, basic coding, good communication). Focus on Aptitude, English proficiency, and foundational logic gates and OS concepts.")


# ==========================================
# PAGE 2: ATS SCANNER
# ==========================================
def page_ats():
    st.image("https://images.unsplash.com/photo-1586281380349-632531db7ed4?auto=format&fit=crop&w=1200&q=80", use_container_width=True)
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
    st.image("https://images.unsplash.com/photo-1506784365847-bbad939e9335?auto=format&fit=crop&w=1200&q=80", use_container_width=True)
    st.title("🗺️ Comprehensive Career Roadmap")
    st.markdown("A deep-dive, 6-month curriculum engineered for your specific target role. Follow the tabs for phase-by-phase execution.")
    
    target_tier = st.selectbox(
        "Select Target Placement Tier:",
        ["Product-Based Companies (SDE)", "Data Analytics & AI Roles", "Service-Based Companies"]
    )
    
    tab1, tab2, tab3 = st.tabs(["Phase 1: Foundation (Months 1-2)", "Phase 2: Core Build (Months 3-4)", "Phase 3: Polish & Interview (Months 5-6)"])
    
    if target_tier == "Product-Based Companies (SDE)":
        with tab1:
            st.markdown("#### Master the Primitives")
            st.write("- **Language Mastery:** Pick Java, C++, or C# and master OOPs (Inheritance, Polymorphism, Abstraction).")
            st.write("- **Basic DSA:** Master Arrays, Strings, Pointers, and Recursion. Understand Time and Space Complexity (Big O).")
            st.write("- **Resource:** Striver's A2Z DSA Sheet or LeetCode Easy problems.")
        with tab2:
            st.markdown("#### Advanced Algorithms & System Design")
            st.write("- **Heavy DSA:** Deep dive into Binary Trees, Graphs, Hashing, and Dynamic Programming.")
            st.write("- **Backend Frameworks:** Learn to build scalable APIs using frameworks like ASP.NET MVC or Node.js.")
            st.write("- **Database:** Master complex SQL queries and understand indexing and normalization.")
        with tab3:
            st.markdown("#### Execution & Mock Interviews")
            st.write("- **Project Deployment:** Push 2 major projects to GitHub with excellent READMEs. Host them live.")
            st.write("- **CS Fundamentals:** Revise Operating Systems (Scheduling, Paging), Computer Networks (TCP/IP, OSI model), and DBMS.")
            st.write("- **Action:** Do mock interviews focusing on communicating your thought process while coding.")

    elif target_tier == "Data Analytics & AI Roles":
        with tab1:
            st.markdown("#### Data Manipulation & Querying")
            st.write("- **Python Fundamentals:** Deep dive into Python. Master list comprehensions and lambda functions.")
            st.write("- **Data Libraries:** Attain fluency in NumPy (array manipulation) and Pandas (DataFrames, merging, cleaning).")
            st.write("- **SQL Mastery:** Window functions, complex joins, CTEs, and aggregate functions.")
        with tab2:
            st.markdown("#### Machine Learning & Visualization")
            st.write("- **Visuals:** Learn Matplotlib and Plotly for interactive chart rendering.")
            st.write("- **ML Algorithms:** Understand Linear/Logistic Regression, Decision Trees, Random Forests, and Time-Series forecasting (ARIMA).")
            st.write("- **Build:** Create an interactive frontend for your models using Streamlit.")
        with tab3:
            st.markdown("#### Portfolio & Business Logic")
            st.write("- **End-to-End Project:** Build a robust, multi-page application (e.g., Stock Market Dashboard or Delivery Time Predictor) and host it.")
            st.write("- **Business Analytics:** Practice guesstimates and business case studies. Learn to translate data into actionable business advice.")
            st.write("- **Action:** Format resume to heavily highlight data cleaning and visualization skills.")

    elif target_tier == "Service-Based Companies":
        with tab1:
            st.markdown("#### Aptitude & Logic First")
            st.write("- **Quantitative Aptitude:** Practice percentages, time and work, speed-distance, and permutations.")
            st.write("- **Logical Reasoning:** Syllogisms, blood relations, and pattern recognition.")
            st.write("- **Basic Coding:** Loops, conditionals, array traversal, and string manipulation in Python or C.")
        with tab2:
            st.markdown("#### Academic Core")
            st.write("- **Subject Revision:** Thoroughly review Operating Systems, Digital Electronics, and Theory of Automata.")
            st.write("- **Web Basics:** Understand the DOM, HTML, CSS, and basic JavaScript interactions.")
            st.write("- **Database:** Basic CRUD operations in SQL.")
        with tab3:
            st.markdown("#### Soft Skills & HR Preparation")
            st.write("- **Communication:** Practice spoken English and articulation. Record yourself answering common HR questions.")
            st.write("- **Resume Polish:** Ensure zero grammatical errors. Highlight any minor college projects or internships.")
            st.write("- **Action:** Prepare a strong 'Tell me about yourself' pitch that lasts exactly 90 seconds.")


# ==========================================
# APP NAVIGATION ROUTING & FLOATING CHATBOT
# ==========================================
st.sidebar.title("System Navigation")
st.sidebar.markdown("**Developer:** Vishal Pandey")
st.sidebar.markdown("**Roll Number:** 2400900100155")
st.sidebar.divider()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

with st.sidebar.popover("💬 Open AI Assistant", use_container_width=True):
    st.markdown("### 🤖 Placement Mentor")
    st.caption("Ask me anything about DSA, interviews, or your resume!")
    
    st.markdown("**Frequently Asked Questions:**")
    faq_col1, faq_col2 = st.columns(2)
    if faq_col1.button("How to improve DSA?"):
        st.session_state.chat_messages.append({"role": "user", "content": "How can I improve my DSA scores for product-based companies?"})
        st.rerun()
    if faq_col2.button("Best projects?"):
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
