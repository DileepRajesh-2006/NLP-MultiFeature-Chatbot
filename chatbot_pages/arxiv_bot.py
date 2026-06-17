

import os
import streamlit as st

os.environ["TRANSFORMERS_VERBOSITY"]       = "error"
os.environ["TOKENIZERS_PARALLELISM"]       = "false"
os.environ["KMP_DUPLICATE_LIB_OK"]        = "TRUE"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import re
import time
import requests
import urllib3
import xml.etree.ElementTree as ET
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
import math

# Suppress SSL warnings (needed for verify=False)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SUMMARIZER_AVAILABLE = False

ARXIV_API_URL = "https://export.arxiv.org/api/query"

ARXIV_TRIGGER_KEYWORDS = [
    "find papers on", "find paper on", "papers on", "paper on",
    "research on", "search for", "search papers", "find research",
    "arxiv", "latest research", "recent papers", "survey on",
    "literature on", "articles on", "publications on",
    "find papers about", "papers about", "research about",
]

VAGUE_QUERIES = {
    "hello", "hi", "hey", "help", "what can you do",
    "what do you do", "ok", "okay", "yes", "no", "thanks",
    "thank you", "cool", "nice", "papers", "paper",
    "research", "search", "find",
}

CONCEPT_EXPLANATIONS = {
    "transformer":            "A Transformer is a deep learning architecture based on self-attention, introduced in 'Attention Is All You Need' (2017). It processes sequences in parallel, making it highly efficient for NLP. Transformers power BERT, GPT, and T5.",
    "attention":              "Attention mechanisms let neural networks focus on relevant parts of the input. Self-attention computes relationships between all positions simultaneously, capturing long-range dependencies without recurrence.",
    "bert":                   "BERT reads text bidirectionally using masked language modeling pre-training. It can be fine-tuned for downstream NLP tasks with minimal labeled data.",
    "gpt":                    "GPT is an autoregressive language model that predicts the next token. It generates coherent text and scales remarkably well with data and compute.",
    "diffusion":              "Diffusion models learn to reverse a gradual noising process. Generation starts from pure noise and iteratively denoises it into a data sample.",
    "embedding":              "Embeddings map discrete objects into a continuous vector space where semantic similarity corresponds to geometric proximity.",
    "neural network":         "Neural networks are computational models organized in layers that learn abstract representations through backpropagation.",
    "reinforcement learning": "RL trains agents to maximize cumulative reward through trial and error, learning a policy that maps states to actions.",
    "convolutional":          "CNNs apply learnable filters across input data to detect local features. Parameter sharing makes them efficient and translation-invariant.",
    "gradient descent":       "Gradient descent iteratively adjusts model parameters in the direction that minimizes a loss function by following the negative gradient.",
    "llm":                    "Large Language Models (LLMs) are transformer-based models trained on massive text corpora. They can generate, summarize, translate, and reason about text with remarkable capability.",
    "gan":                    "Generative Adversarial Networks (GANs) consist of a generator and discriminator trained adversarially. The generator creates realistic samples while the discriminator learns to distinguish real from fake.",
    "lstm":                   "Long Short-Term Memory networks are recurrent neural networks with gating mechanisms to retain long-range dependencies, widely used before transformers replaced them for sequence tasks.",
}

CONCEPT_RELATIONS = {
    "transformer":            ["attention", "bert", "gpt", "embedding", "neural network"],
    "attention":              ["transformer", "bert", "gpt", "neural network"],
    "bert":                   ["transformer", "attention", "embedding", "neural network"],
    "gpt":                    ["transformer", "attention", "embedding", "neural network", "llm"],
    "llm":                    ["gpt", "bert", "transformer", "embedding"],
    "neural network":         ["deep learning", "gradient descent", "embedding", "convolutional"],
    "deep learning":          ["neural network", "convolutional", "transformer", "gradient descent"],
    "convolutional":          ["neural network", "deep learning"],
    "gradient descent":       ["neural network", "deep learning"],
    "embedding":              ["transformer", "bert", "gpt", "neural network"],
    "reinforcement learning": ["neural network", "deep learning"],
    "diffusion":              ["neural network", "deep learning", "gan"],
    "gan":                    ["neural network", "deep learning", "diffusion"],
    "lstm":                   ["neural network", "deep learning", "transformer"],
}


# ── Cache at module level — results cached 5 min to avoid repeated requests ──
def search_arxiv(query, max_results=5):
    """Search arXiv with retry + exponential backoff."""
    params = {
        "search_query": f"all:{query}",
        "start":        0,
        "max_results":  max_results,
        "sortBy":       "relevance",
        "sortOrder":    "descending",
    }
    for attempt in range(3):
        try:
            resp = requests.get(
                ARXIV_API_URL,
                params=params,
                timeout=45,
                verify=False        # ✅ bypass SSL cert issue on Windows
            )
            if resp.status_code == 429:
                wait = 20 * (attempt + 1)   # 20s, 40s, 60s
                time.sleep(wait)
                continue
            if resp.status_code == 403:
                return [], "BLOCKED"
            resp.raise_for_status()
            break
        except requests.exceptions.ConnectionError:
            return [], "CONNECTION_ERROR"
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(5)
                continue
            return [], "TIMEOUT"
        except Exception as e:
            return [], str(e)
    else:
        return [], "RATE_LIMITED"

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return [], "PARSE_ERROR"

    papers = []
    for entry in root.findall("atom:entry", ns):
        def t(tag):
            el = entry.find(f"atom:{tag}", ns)
            return el.text.strip() if el is not None and el.text else ""
        raw_id = t("id")
        authors = [
            a.find("atom:name", ns).text.strip()
            for a in entry.findall("atom:author", ns)
            if a.find("atom:name", ns) is not None
        ]
        categories = [c.get("term", "") for c in entry.findall("atom:category", ns)]
        papers.append({
            "title":      t("title").replace("\n", " "),
            "abstract":   t("summary").replace("\n", " "),
            "authors":    authors,
            "published":  t("published")[:10],
            "url":        raw_id,
            "pdf_url":    raw_id.replace("/abs/", "/pdf/") + ".pdf",
            "categories": categories,
        })
    return papers, None




def show():

    for key, default in {
        "arxiv_chat_history": [],
        "arxiv_active_paper": None,
        "arxiv_last_results": [],
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default



    def build_concept_graph(concepts):
        if not concepts:
            return None
        nodes = list(set(concepts))
        n = len(nodes)
        edges = []
        for i, c1 in enumerate(nodes):
            for j, c2 in enumerate(nodes):
                if i >= j:
                    continue
                relations = CONCEPT_RELATIONS.get(c1.lower(), [])
                if c2.lower() in relations:
                    edges.append((i, j))
                elif any(
                    c1.lower() in CONCEPT_RELATIONS.get(k, [])
                    and c2.lower() in CONCEPT_RELATIONS.get(k, [])
                    for k in CONCEPT_RELATIONS
                ):
                    edges.append((i, j))
        angle_step = 2 * math.pi / max(n, 1)
        node_x = [math.cos(i * angle_step) for i in range(n)]
        node_y = [math.sin(i * angle_step) for i in range(n)]
        edge_x, edge_y = [], []
        for i, j in edges:
            edge_x += [node_x[i], node_x[j], None]
            edge_y += [node_y[i], node_y[j], None]
        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
            "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
            "#BB8FCE", "#F1948A",
        ]
        fig = go.Figure(
            data=[
                go.Scatter(x=edge_x, y=edge_y, mode="lines",
                           line=dict(width=1.5, color="#888"), hoverinfo="none"),
                go.Scatter(x=node_x, y=node_y, mode="markers+text",
                           text=nodes, textposition="top center", hoverinfo="text",
                           marker=dict(size=30, color=colors[:n],
                                       line=dict(width=2, color="white"))),
            ],
            layout=go.Layout(
                title=dict(text="🧠 Concept Relationship Network", font=dict(size=16)),
                showlegend=False, hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=400,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
        )
        return fig

    def build_timeline_chart(papers):
        if not papers:
            return None
        titles = [p["title"][:30] + "..." for p in papers]
        years  = [p["published"][:4] for p in papers]
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=titles, y=years,
            marker_color=colors[:len(titles)],
            text=years, textposition="auto"
        ))
        fig.update_layout(
            title="📅 Paper Publication Years",
            xaxis_title="Paper", yaxis_title="Year", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    def build_category_pie(papers):
        all_cats = [c for p in papers for c in p.get("categories", [])]
        if not all_cats:
            return None
        cat_counts = Counter(all_cats).most_common(6)
        fig = px.pie(
            values=[c[1] for c in cat_counts],
            names=[c[0] for c in cat_counts],
            title="📂 Category Distribution", hole=0.3,
        )
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
        return fig



    def extract_topic(text):
        text_lower = text.lower()
        for phrase in sorted(ARXIV_TRIGGER_KEYWORDS, key=len, reverse=True):
            if phrase in text_lower:
                topic = text_lower.replace(phrase, "").strip().strip("?.")
                if len(topic) > 1:
                    return topic
        return text.strip()

    def extract_concepts_from_text(text):
        concept_keywords = [
            "neural network", "deep learning", "transformer", "attention",
            "reinforcement learning", "generative", "classification",
            "regression", "clustering", "optimization", "gradient",
            "embedding", "encoder", "decoder", "convolution", "dropout",
            "transfer learning", "fine-tuning", "pre-training",
            "bert", "gpt", "diffusion", "gan", "vae", "llm",
            "convolutional", "recurrent", "lstm", "graph neural",
        ]
        text_lower = text.lower()
        return list(set(kw for kw in concept_keywords if kw in text_lower))[:8]

    def simple_summary(text, n=2):
        stopwords = {
            "the","a","an","in","of","on","and","or","for","to","with","is","are",
            "was","were","that","this","it","as","by","from","be","at","we","our",
            "have","has","which","their","these","can","also","not","but","more",
        }
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) <= n:
            return text
        words_all = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        freq = Counter(w for w in words_all if w not in stopwords)
        keywords = set(w for w, _ in freq.most_common(15))
        scores = [
            len(set(re.findall(r'\b[a-zA-Z]{4,}\b', s.lower())) & keywords)
            for s in sentences
        ]
        top = sorted(
            sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:n]
        )
        return " ".join(sentences[i] for i in top)

    def summarize(abstract):
        return simple_summary(abstract, n=2)

    def explain_concept(concept):
        try:
            payload = {
                "model":  "mistral",
                "prompt": f"Explain '{concept}' in 3 sentences for a CS graduate student.",
                "stream": False,
            }
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json=payload, timeout=10
            )
            if resp.status_code == 200:
                answer = resp.json().get("response", "").strip()
                if answer:
                    return answer
        except Exception:
            pass
        for key, explanation in CONCEPT_EXPLANATIONS.items():
            if key in concept.lower():
                return explanation
        return (
            f"**{concept}** is a key concept in CS research. "
            "Search arXiv for foundational papers on this topic."
        )



    def handle_message(user_input):
        text_lower = user_input.lower().strip()
        active  = st.session_state.arxiv_active_paper
        results = st.session_state.arxiv_last_results

        if any(w in text_lower for w in ["summarize", "summary", "summarise", "tldr", "brief"]):
            if active:
                with st.spinner("Summarizing..."):
                    s = summarize(active["abstract"])
                return (
                    f"📄 **Summary of '{active['title'][:65]}...'**\n\n"
                    f"{s}\n\n"
                    f"*{active['published']} · {', '.join(active['authors'][:2])}*"
                )
            return "No active paper yet. Try: **find papers on transformers**"

        if any(w in text_lower for w in ["contribution", "key contribution", "key idea", "findings", "finding"]):
            if active:
                s = simple_summary(active["abstract"], n=3)
                return f"🔑 **Key Contributions:**\n\n{s}"
            return "No active paper. Search for papers first."

        if "author" in text_lower:
            if active:
                return "👥 **Authors:** " + ", ".join(active["authors"])
            return "No active paper. Search for papers first."

        if any(w in text_lower for w in ["pdf", "link", "download"]):
            if active:
                return f"📎 **PDF:** [{active['title'][:60]}...]({active['pdf_url']})"
            return "No active paper. Search for papers first."

        if any(w in text_lower for w in ["category", "categories", "field", "area"]):
            if active:
                return f"🏷️ **Categories:** {', '.join(active['categories'])}"
            return "No active paper. Search for papers first."

        if any(w in text_lower for w in [
            "list papers", "show papers", "what papers",
            "loaded papers", "available papers", "papers available", "papers loaded"
        ]):
            if results:
                lines = [f"📋 **{len(results)} papers currently loaded:**\n"]
                for i, p in enumerate(results, 1):
                    lines.append(f"**{i}.** {p['title']} *({p['published']})*")
                lines.append("\nType `summarize`, `authors`, `pdf link`, or `key contributions` to learn more.")
                return "\n".join(lines)
            return "No papers loaded yet. Try: **find papers on transformers**"

        if any(w in text_lower for w in ["explain", "what is", "what are", "define", "how does", "describe"]):
            concept = re.sub(
                r"\b(explain|what|is|are|define|definition|of|how|does|the|a|an|its|their|this|that|it)\b",
                " ", text_lower
            ).strip().strip("?")
            concept = re.sub(r"\s+", " ", concept).strip()
            if concept and len(concept) > 2:
                return f"💡 **{concept.title()}**\n\n{explain_concept(concept)}"
            if active:
                s = simple_summary(active["abstract"], n=2)
                return (
                    f"Here's what the active paper is about:\n\n{s}\n\n"
                    "To explain a concept, try: `explain transformer` or `what is attention`"
                )
            return (
                "Please specify a concept, e.g.:\n"
                "- `explain transformer`\n"
                "- `what is attention`\n"
                "- `how does BERT work`"
            )

        if text_lower in VAGUE_QUERIES or len(text_lower) < 4:
            if active:
                return (
                    f"📌 Active paper: **{active['title'][:70]}...**\n\n"
                    "Ask me to: `summarize` · `explain [concept]` · `authors` · `pdf link` · `key contributions`\n\n"
                    "Or search: **find papers on [topic]**"
                )
            return (
                "👋 I'm your arXiv Research Assistant!\n\n"
                "Try: **find papers on transformers**\n"
                "Then ask: `summarize` · `explain attention` · `authors` · `pdf link`"
            )

        # Paper search
        topic = extract_topic(user_input)
        with st.spinner(f"🔍 Searching arXiv for **{topic}**..."):
            papers, err = search_arxiv(topic, max_results=5)

        if err == "CONNECTION_ERROR":
            return "❌ Cannot connect to arXiv. Check your internet connection."
        if err == "TIMEOUT":
            return "⏱ arXiv timed out. Please try again."
        if err == "RATE_LIMITED":
            return "⏳ arXiv rate limit hit after 3 retries. Please wait 1-2 minutes and try again."
        if err == "BLOCKED":
            return "❌ arXiv blocked the request. Try again in a minute."
        if err:
            return f"❌ arXiv error: `{err}`"
        if not papers:
            return f"🔍 No papers found for **{topic}**. Try a different search term."

        st.session_state.arxiv_active_paper = papers[0]
        st.session_state.arxiv_last_results = papers

        lines = [f"📚 Found **{len(papers)} papers** on **{topic}**:\n"]
        for i, p in enumerate(papers, 1):
            s = simple_summary(p["abstract"], n=1)
            authors_str = ", ".join(p["authors"][:2])
            if len(p["authors"]) > 2:
                authors_str += " et al."
            lines.append(
                f"**{i}. {p['title']}**\n"
                f"*{p['published']} · {authors_str}*\n"
                f"{s}\n"
                f"[📎 PDF]({p['pdf_url']})\n"
            )
        lines.append(
            "---\n"
            "💡 Paper 1 is now active. Ask me to:\n"
            "`summarize` · `explain [concept]` · `authors` · `pdf link` · `key contributions`"
        )
        return "\n".join(lines)



    st.title("🔬 arXiv Research Assistant")
    st.markdown(
        "Search for CS research papers and chat about them. "
        "Type a topic to search, or ask follow-up questions about the active paper."
    )

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 💡 How to use")
        st.markdown(
            "**Search:** Type a topic directly in chat\n\n"
            "**Examples:**\n"
            "- `find papers on transformers`\n"
            "- `papers on diffusion models`\n"
            "- `research on reinforcement learning`\n\n"
            "**Follow-ups after search:**\n"
            "- `summarize`\n"
            "- `explain attention`\n"
            "- `authors`\n"
            "- `pdf link`\n"
            "- `key contributions`"
        )
        active = st.session_state.arxiv_active_paper
        if active:
            st.markdown("---")
            st.markdown("### 📌 Active Paper")
            st.caption(active["title"][:80] + "…")
            st.caption(f"📅 {active['published']}")
            st.caption(f"🏷️ {', '.join(active['categories'][:2])}")
            if st.button("❌ Clear Active Paper"):
                st.session_state.arxiv_active_paper = None
                st.session_state.arxiv_last_results = []
                st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.arxiv_chat_history = []
        st.session_state.arxiv_active_paper = None
        st.session_state.arxiv_last_results = []
        st.rerun()

    st.markdown("---")

    results = st.session_state.arxiv_last_results
    active  = st.session_state.arxiv_active_paper

    if results or active:
        st.subheader("📊 Visualizations")
        viz_tab1, viz_tab2 = st.tabs(["🧠 Concept Network", "📅 Paper Timeline"])

        with viz_tab1:
            all_text = ""
            if active:
                all_text += active["abstract"]
            for p in results:
                all_text += " " + p["abstract"]
            concepts = extract_concepts_from_text(all_text)
            if concepts:
                fig = build_concept_graph(concepts)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Detected Concepts:**")
                badge_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
                                "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"]
                cols = st.columns(min(len(concepts), 4))
                for i, concept in enumerate(concepts):
                    with cols[i % 4]:
                        st.markdown(
                            f"<div style='background:{badge_colors[i % len(badge_colors)]};"
                            f"padding:6px 10px;border-radius:20px;text-align:center;"
                            f"font-size:12px;font-weight:bold;color:white;margin:4px'>"
                            f"{concept}</div>",
                            unsafe_allow_html=True
                        )
            else:
                st.info("Search for papers to see concept visualization.")

        with viz_tab2:
            if results:
                fig2 = build_timeline_chart(results)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
                fig3 = build_category_pie(results)
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Search for papers to see timeline and category charts.")

        st.markdown("---")

    if not st.session_state.arxiv_chat_history:
        with st.chat_message("assistant"):
            st.markdown(
                "👋 Hello! I'm your **arXiv Research Assistant**.\n\n"
                "Just type a topic to search for papers:\n\n"
                "- `find papers on transformers`\n"
                "- `papers on large language models`\n"
                "- `research on diffusion models`\n\n"
                "After a search, ask me to `summarize`, `explain` a concept, "
                "list `authors`, or get the `pdf link`."
            )

    for msg in st.session_state.arxiv_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("source"):
                st.markdown(
                    f"<small style='color:gray'>📌 Source: {msg['source']}</small>",
                    unsafe_allow_html=True
                )

    user_input = st.chat_input(
        "e.g. find papers on transformers / summarize / explain attention"
    )

    if user_input:
        user_input = user_input.strip()
        st.session_state.arxiv_chat_history.append({
            "role": "user", "content": user_input,
        })
        response = handle_message(user_input)
        st.session_state.arxiv_chat_history.append({
            "role":    "assistant",
            "content": response,
            "source":  "arXiv Research Assistant",
        })
        st.rerun()