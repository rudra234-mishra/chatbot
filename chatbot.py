import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import AzureChatOpenAI

load_dotenv()

st.set_page_config(
    page_title="Rudra QnA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #1d4ed8 0%, #0f172a 35%, #020617 100%);
    color: #e5e7eb;
}

[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: rgba(0,0,0,0);
}

.main .block-container {
    padding-top: 2rem;
    max-width: 900px;
}

h1, p, label, span, div {
    color: #e5e7eb;
}

.stChatMessage {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 0.25rem 0.5rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.stChatInput input {
    background: rgba(255,255,255,0.08) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 999px !important;
}

.stChatInput input::placeholder {
    color: #cbd5e1 !important;
}

[data-testid="stChatInput"] {
    background: rgba(15, 23, 42, 0.75);
    border-radius: 999px;
    padding: 0.35rem;
    border: 1px solid rgba(255,255,255,0.12);
}

.stButton button {
    border-radius: 999px;
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Rudra QnA System")
st.markdown("Ask anything and get answers powered by Azure OpenAI.")

@st.cache_resource
def load_model():
    return AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("api_version"),
        azure_deployment=os.getenv("AZURE_OPENAI_MODEL"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        streaming=True,
        temperature=0.7,
    )

model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask your question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for chunk in model.stream(prompt):
            if chunk.content:
                full_response += chunk.content
                placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})