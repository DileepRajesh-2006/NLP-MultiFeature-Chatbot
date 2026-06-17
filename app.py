# ============================================================
# MAIN STREAMLIT APP
# Unified AI Chatbot Suite
# ============================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"]      = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"]     = "false"
os.environ["TRANSFORMERS_VERBOSITY"]     = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
import streamlit as st

st.set_page_config(
    page_title="AI Chatbot Suite",
    page_icon="🤖",
    layout="wide"
)

st.sidebar.title("🤖 AI Chatbot Suite")
st.sidebar.markdown("---")
st.sidebar.markdown("### Standalone Bots")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "🤖 Main Chatbot",
        "🏥 Medical Q&A (Task 2)",
        "📚 Dynamic Knowledge Bot (Task 3)",
        "🔬 arXiv Research Assistant (Task 4)",
        "🎨 Multimodal Chatbot (Task 5)",
        "🌐 Multilingual Chatbot (Task 6)",
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ using Python & Streamlit")

if page == "🤖 Main Chatbot":
    from chatbot_pages import main_chatbot
    main_chatbot.show()

elif page == "🏥 Medical Q&A (Task 2)":
    from chatbot_pages import medical_bot
    medical_bot.show()

elif page == "📚 Dynamic Knowledge Bot (Task 3)":
    from chatbot_pages import dynamic_knowledge_bot
    dynamic_knowledge_bot.show()

elif page == "🔬 arXiv Research Assistant (Task 4)":
    from chatbot_pages import arxiv_bot
    arxiv_bot.show()

elif page == "🎨 Multimodal Chatbot (Task 5)":
    from chatbot_pages import multimodal_bot
    multimodal_bot.show()

elif page == "🌐 Multilingual Chatbot (Task 6)":
    from chatbot_pages import multilingual_bot
    multilingual_bot.show()