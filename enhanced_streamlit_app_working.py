"""
Enhanced Streamlit UI for Intelligent Document Q&A System
Complete interface with document upload, chat, analytics, and advanced features
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import io
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="Intelligent Document Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        margin-left: 2rem;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        margin-right: 2rem;
    }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        background-color: #f8f9ff;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
def init_session_state():
    """Initialize all session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'token' not in st.session_state:
        st.session_state.token = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"streamlit_{int(time.time())}"

# API helper functions
def make_api_request(endpoint, method="GET", data=None, files=None, token=None):
    """Make API request with proper error handling"""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            if files:
                headers.pop("Content-Type", None)  # Let requests set it for files
                response = requests.post(url, files=files, headers=headers)
            else:
                response = requests.post(url, json=data, headers=headers)
        
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None

def check_api_health():
    """Check if API server is running"""
    response = make_api_request("/health")
    return response and response.status_code == 200

def login_user(username, password):
    """Authenticate user with API"""
    response = make_api_request("/auth/login", "POST", {"username": username, "password": password})
    
    if response and response.status_code == 200:
        data = response.json()
        st.session_state.authenticated = True
        st.session_state.token = data["access_token"]
        st.session_state.username = data["username"]
        st.session_state.user_role = data["role"]
        return True
    return False

def logout_user():
    """Logout and clear session"""
    st.session_state.authenticated = False
    st.session_state.token = None
    st.session_state.username = None
    st.session_state.user_role = None
    st.session_state.chat_history = []

# Authentication interface
def render_login_page():
    """Render the login interface"""
    st.markdown('<div class="main-header"><h1>🤖 Intelligent Document Q&A System</h1><p>Your AI-powered document assistant</p></div>', unsafe_allow_html=True)
    
    # Check API health first
    if not check_api_health():
        st.error("⚠️ **API server is not running!**")
        st.info("Please start the server first:")
        st.code("uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000")
        return
    
    st.success("✅ API server is running and healthy")
    
    # Login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Please Login")
        
        with st.form("login_form"):
            username = st.text_input("Username", value="admin", help="Default: admin")
            password = st.text_input("Password", type="password", value="admin123", help="Default: admin123")
            
            col_a, col_b = st.columns(2)
            with col_a:
                login_btn = st.form_submit_button("🚀 Login", use_container_width=True)
            with col_b:
                demo_btn = st.form_submit_button("👀 Demo Mode", use_container_width=True)
            
            if login_btn:
                with st.spinner("Authenticating..."):
                    if login_user(username, password):
                        st.success("Login successful! Welcome!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Try admin/admin123")
            
            if demo_btn:
                if login_user("admin", "admin123"):
                    st.success("Demo mode activated!")
                    time.sleep(1)
                    st.rerun()

# Upload function
def upload_documents(files):
    """Upload documents via API"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    success_count = 0
    
    for i, file in enumerate(files):
        status_text.text(f"Uploading {file.name}...")
        
        # Reset file pointer to beginning
        file.seek(0)
        
        # Prepare file data
        file_content = file.read()
        files_data = {'file': (file.name, io.BytesIO(file_content), file.type)}
        
        response = make_api_request(
            "/documents/upload",
            "POST",
            files=files_data,
            token=st.session_state.token
        )
        
        if response and response.status_code == 200:
            result = response.json()
            st.success(f"✅ **{file.name}** uploaded successfully! Created {result['chunks_created']} chunks in {result['processing_time']:.2f}s")
            success_count += 1
        else:
            error_msg = "Unknown error"
            if response:
                try:
                    error_data = response.json()
                    error_msg = error_data.get('detail', response.text)
                except:
                    error_msg = response.text
            st.error(f"❌ Failed to upload **{file.name}**: {error_msg}")
        
        progress_bar.progress((i + 1) / len(files))
    
    status_text.text(f"Upload complete! {success_count}/{len(files)} files processed successfully.")
    
    if success_count > 0:
        st.balloons()

# Chat functions
def ask_question(question):
    """Process a question through the Q&A API"""
    with st.spinner("🤔 AI is thinking..."):
        response = make_api_request(
            "/qa/ask",
            "POST",
            {
                "question": question,
                "session_id": st.session_state.session_id
            },
            token=st.session_state.token
        )
        
        if response and response.status_code == 200:
            data = response.json()
            
            # Add to chat history
            st.session_state.chat_history.append((
                question,
                data["answer"],
                data["sources"],
                data["confidence_score"],
                datetime.now().strftime("%H:%M")
            ))
            
            st.success("✅ Response generated!")
            st.rerun()
        else:
            st.error("❌ Failed to get response. Please try again.")

# Main application interface
def render_main_app():
    """Render the main application"""
    
    # Header with user info and logout
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f'<div class="main-header"><h2>🤖 Document Q&A Dashboard</h2><p>Welcome back, {st.session_state.username}!</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f"**Role:** {st.session_state.user_role}")
        st.markdown(f"**Session:** {st.session_state.session_id[-8:]}")
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.rerun()
    
    # Navigation tabs
    tab1, tab2 = st.tabs(["💬 Chat & Upload", "📚 Document Manager"])
    
    with tab1:
        render_chat_page()
    
    with tab2:
        render_documents_page()

def render_chat_page():
    """Render the chat and Q&A interface"""
    st.header("💬 Chat with Your Documents")
    
    # Main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat history display
        st.markdown("### 🗨️ Conversation")
        
        if st.session_state.chat_history:
            for i, (question, answer, sources, confidence, timestamp) in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🙋 You ({timestamp}):</strong><br>
                    {question}
                </div>
                """, unsafe_allow_html=True)
                
                # Bot response
                confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 AI Assistant:</strong><br>
                    {answer}<br>
                    <small>{confidence_color} Confidence: {confidence:.2f} | 📚 Sources: {len(sources)}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources in expandable section
                if sources:
                    with st.expander(f"📖 View Sources for Response {i+1}"):
                        for j, source in enumerate(sources):
                            st.markdown(f"**Source {j+1}:**")
                            st.write(source.get('content', 'No content available'))
                            st.markdown("---")
        else:
            st.info("👋 Welcome! Upload documents and start asking questions.")
        
        # Question input area
        st.markdown("### ✍️ Ask a Question")
        
        with st.form("question_form", clear_on_submit=True):
            question = st.text_area(
                "What would you like to know?",
                placeholder="e.g., What is artificial intelligence? How does machine learning work?",
                height=80
            )
            
            col_a, col_b = st.columns([3, 1])
            with col_a:
                ask_btn = st.form_submit_button("🚀 Ask Question", use_container_width=True)
            with col_b:
                clear_btn = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)
            
            if ask_btn and question.strip():
                ask_question(question)
            
            if clear_btn:
                st.session_state.chat_history = []
                st.success("Chat cleared!")
                st.rerun()
    
    with col2:
        # Quick upload section
        st.markdown("### 📤 Quick Upload")
        
        # Inline file uploader for immediate document access
        quick_upload_files = st.file_uploader(
            "Upload documents from your laptop:",
            type=['txt', 'md', 'pdf', 'docx'],
            accept_multiple_files=True,
            help="Select documents to upload and chat with immediately",
            key="quick_upload"
        )
        
        if quick_upload_files:
            st.markdown("**Files selected:**")
            for file in quick_upload_files:
                st.write(f"📄 {file.name} ({file.size:,} bytes)")
            
            if st.button("🚀 Upload & Start Chatting", use_container_width=True, type="primary"):
                upload_documents(quick_upload_files)
                st.success("✅ Documents uploaded! You can now ask questions about them below.")
        
        st.markdown("---")
        
        # Sample questions
        st.markdown("### 💡 Sample Questions")
        sample_questions = [
            "What is artificial intelligence?",
            "How does machine learning work?", 
            "Explain neural networks",
            "What are the types of AI?",
        ]
        
        for sq in sample_questions:
            if st.button(f"💬 {sq}", key=f"sample_{sq[:15]}", use_container_width=True):
                ask_question(sq)

def render_documents_page():
    """Render document management interface"""
    st.header("📚 Document Management")
    
    # Prominent upload section
    st.markdown("""
    <div class="upload-area">
        <h3>📤 Upload Documents from Your Laptop</h3>
        <p>Select documents from your computer to upload and chat with them instantly!</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose documents to upload",
        type=['txt', 'md', 'pdf', 'docx'],
        accept_multiple_files=True,
        help="Supported formats: TXT, Markdown, PDF, DOCX. You can select multiple files at once.",
        key="main_upload"
    )
    
    if uploaded_files:
        st.markdown("#### 📋 Files Selected for Upload:")
        
        # Display files in a nice table format
        file_data = []
        total_size = 0
        
        for file in uploaded_files:
            file_data.append({
                "📄 Filename": file.name,
                "📐 Size": f"{file.size:,} bytes",
                "📝 Type": file.type or "Unknown",
                "✅ Status": "Ready to upload"
            })
            total_size += file.size
        
        df = pd.DataFrame(file_data)
        st.dataframe(df, use_container_width=True)
        
        st.info(f"📊 **Summary:** {len(uploaded_files)} files selected, total size: {total_size:,} bytes")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button("🚀 Upload All Documents", use_container_width=True, type="primary"):
                upload_documents(uploaded_files)
        with col3:
            if st.button("🗑️ Clear Selection", use_container_width=True):
                st.rerun()
    
    # Document library placeholder
    st.markdown("### 📚 Document Library")
    st.info("Documents you upload will appear here after processing.")

# Main application
def main():
    """Main application entry point"""
    init_session_state()
    
    if not st.session_state.authenticated:
        render_login_page()
    else:
        render_main_app()

if __name__ == "__main__":
    main()
