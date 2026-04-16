# Mehreen Irfan
# Streamlit Web UI for Medical RAG System

import streamlit as st
import os
from data_retrieval import HybridRetriever, load_data
from RAG import RAG_query

# Page configuration
st.set_page_config(
    page_title="DoctorFollow AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 20px;
    }
    .query-box {
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    .result-box {
        padding: 15px;
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 10px 0;
    }
    .article-box {
        padding: 12px;
        background-color: #fff9e6;
        border-left: 4px solid #ff9800;
        border-radius: 5px;
        margin: 8px 0;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>🏥 DoctorFollow AI</h1>
        <p><em>Medical Research-Augmented Generation System</em></p>
        <p style="font-size: 0.9em; color: gray;">Query medical literature from PubMed with AI-powered answers</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration")
top_n = st.sidebar.slider("Number of articles to retrieve:", 1, 10, 5)
language_info = st.sidebar.info(
    "🌐 **Language**: Queries are auto-detected. "
    "Turkish questions get Turkish answers. English questions get English answers."
)

api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if api_key or groq_api_key:
    st.sidebar.success("✅ LLM API keys configured (Gemini/Groq available)")
else:
    st.sidebar.warning("⚠️ No API keys set. Set GOOGLE_API_KEY or GROQ_API_KEY for answer generation.")

# Load data once (caching)
@st.cache_resource
def load_system():
    """Load articles and initialize retriever once"""
    try:
        articles = load_data("data/pubmed_refr.json")
        retriever = HybridRetriever(articles)
        return articles, retriever
    except FileNotFoundError:
        return None, None

articles, retriever = load_system()

if articles is None or retriever is None:
    st.error("❌ Error: data/pubmed_refr.json not found. Run `python main.py` first to fetch articles.")
    st.stop()

# Main UI
col1, col2 = st.columns([3, 1], gap="medium")

with col1:
    st.markdown('<div class="query-box">', unsafe_allow_html=True)
    st.subheader("📝 Ask a Medical Question")
    query = st.text_area(
        "Enter your query (Turkish or English):",
        placeholder="Example: Iron supplementation for anemia during pregnancy\nOr: Çocuklarda akut otitis media tedavisi",
        height=100
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.subheader("📊 Statistics")
    st.metric("Articles Indexed", len(articles))
    st.metric("Retrieval Method", "Hybrid (BM25 + Semantic)")

# Process query
if st.button("🔍 Search & Generate Answer", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("⏳ Retrieving articles and generating answer..."):
            try:
                result = RAG_query(retriever, query, top_n=top_n, api_key=api_key)
                
                # Display results
                st.markdown("---")
                st.subheader("📖 Retrieved Articles")
                
                # Articles table
                article_data = []
                for article in result["retrieved_articles"]:
                    article_data.append({
                        "Rank": article["rank"],
                        "PMID": article["pmid"],
                        "Title": article["title"][:60] + "..." if len(article["title"]) > 60 else article["title"],
                        "Score": f"{article['score']:.3f}"
                    })
                
                st.dataframe(article_data, use_container_width=True, hide_index=True)
                
                # Full article details (expandable)
                with st.expander("📚 View Full Article Details"):
                    for article in result["retrieved_articles"]:
                        st.markdown(f"""
                        <div class="article-box">
                            <b>PMID: {article['pmid']}</b><br>
                            <b>Title:</b> {article['title']}<br>
                            <b>Relevance Score:</b> {article['score']:.4f}
                        </div>
                        """, unsafe_allow_html=True)
                
                # LLM Answer
                st.markdown("---")
                st.subheader("🤖 Generated Answer")
                st.markdown(f"""
                <div class="result-box">
                {result['answer']}
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: gray; font-size: 0.85em; padding: 20px;">
        <p>DoctorFollow AI — Medical RAG System | Built with Streamlit</p>
        <p>Requires: GOOGLE_API_KEY or GROQ_API_KEY environment variable for LLM features</p>
    </div>
""", unsafe_allow_html=True)
