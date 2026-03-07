import streamlit as st
import os
import json
import tempfile
from rag_pipeline import build_vectorstore, full_contract_analysis, ask_question

st.set_page_config(page_title="⚖️ Legal AI Assistant", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-title { font-family: 'DM Serif Display', serif; font-size: 2.8rem; color: #1a1a2e; }
    .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
    .card-title { font-family: 'DM Serif Display', serif; font-size: 1.1rem; color: #1a1a2e; margin-bottom: 0.6rem; border-bottom: 2px solid #f3f4f6; padding-bottom: 0.4rem; }
    .risk-high { background: #fef2f2; border-left: 4px solid #ef4444; padding: 0.6rem 1rem; border-radius: 0 8px 8px 0; margin: 0.4rem 0; }
    .risk-medium { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 0.6rem 1rem; border-radius: 0 8px 8px 0; margin: 0.4rem 0; }
    .risk-low { background: #f0fdf4; border-left: 4px solid #22c55e; padding: 0.6rem 1rem; border-radius: 0 8px 8px 0; margin: 0.4rem 0; }
    .badge { display: inline-block; padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.75rem; font-weight: 500; }
    .badge-blue { background: #dbeafe; color: #1d4ed8; }
    .badge-green { background: #dcfce7; color: #15803d; }
    .badge-yellow { background: #fef9c3; color: #a16207; }
    .badge-red { background: #fee2e2; color: #b91c1c; }
    .tag { display: inline-block; background: #f3f4f6; color: #374151; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.8rem; margin: 0.2rem; font-family: 'DM Mono', monospace; }
    .summary-box { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #e2e8f0; padding: 1.5rem; border-radius: 12px; font-size: 0.95rem; line-height: 1.7; margin-bottom: 1.5rem; }
    .json-box { background: #1e1e2e; color: #cdd6f4; padding: 1rem; border-radius: 10px; font-family: 'DM Mono', monospace; font-size: 0.78rem; overflow-x: auto; max-height: 400px; overflow-y: auto; }
    .stButton > button { background: #1a1a2e !important; color: white !important; border: none; border-radius: 8px; font-weight: 500; }
    .stButton > button:hover { background: #16213e !important; transform: translateY(-1px); }
    .confidence-high { color: #15803d; font-weight: 600; }
    .confidence-medium { color: #a16207; font-weight: 600; }
    .confidence-low { color: #b91c1c; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    groq_key = st.text_input("Groq API Key", type="password",
                              help="Get free key at console.groq.com")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    model_choice = st.selectbox(
        "Groq Model",
        ["llama3-70b-8192", "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        help="llama3-8b = fastest | llama3-70b = most accurate"
    )
    st.divider()
    st.markdown("🤗 **Embeddings:** HuggingFace `all-MiniLM-L6-v2`")
    st.markdown("⚡ **LLM:** Groq LLaMA3 — FREE")
    st.markdown("📐 **Output:** Pydantic JSON Schema")
    st.markdown("[Get free Groq key →](https://console.groq.com)")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">⚖️ Legal AI Assistant</p>', unsafe_allow_html=True)
st.caption("HuggingFace Embeddings · Groq LLaMA3 · Structured JSON Output · RAG")
st.divider()

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📄 Upload Legal Contract (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file and groq_key:
    with tempfile.NamedTemporaryFile(delete=False,
            suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("🔄 Chunking → HuggingFace Embeddings → FAISS..."):
        vectorstore = build_vectorstore(tmp_path)
        st.session_state["vectorstore"] = vectorstore
        st.session_state["model"] = model_choice

    st.success(f"✅ **{uploaded_file.name}** indexed successfully!")

elif uploaded_file and not groq_key:
    st.warning("⚠️ Enter your Groq API key in the sidebar.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
if "vectorstore" in st.session_state:
    tab1, tab2, tab3 = st.tabs(["🔍 Full Analysis", "💬 Ask a Question", "🗂️ Raw JSON"])

    # ── TAB 1 ─────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("### 📋 Complete Contract Analysis")
        st.caption("Extracts all structured data using JSON schema validation")

        if st.button("🚀 Run Full Analysis", type="primary", use_container_width=True):
            with st.spinner("⚡ Analyzing + parsing structured output..."):
                structured, raw, sources, error = full_contract_analysis(
                    st.session_state["vectorstore"], st.session_state["model"]
                )
                st.session_state["full_analysis"] = structured
                st.session_state["full_analysis_raw"] = raw
                st.session_state["full_analysis_error"] = error

        if "full_analysis" in st.session_state:
            analysis = st.session_state["full_analysis"]
            error = st.session_state["full_analysis_error"]

            if error:
                st.error(f"Parsing error: {error}")
                st.text(st.session_state["full_analysis_raw"])
            elif analysis:
                st.markdown(
                    f'<div class="summary-box">📝 <strong>Summary</strong><br><br>{analysis.summary}</div>',
                    unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="card"><div class="card-title">👥 Parties</div>', unsafe_allow_html=True)
                    for p in analysis.parties:
                        st.markdown(f'<span class="badge badge-blue">{p.role}</span> &nbsp; {p.name}', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="card"><div class="card-title">📅 Key Dates</div>', unsafe_allow_html=True)
                    if analysis.key_dates:
                        for d in analysis.key_dates:
                            st.markdown(f'<span class="tag">{d.date}</span> &nbsp; {d.event}', unsafe_allow_html=True)
                    else:
                        st.caption("No specific dates found")
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="card"><div class="card-title">💰 Payment Terms</div>', unsafe_allow_html=True)
                    if analysis.payment_terms:
                        for pt in analysis.payment_terms:
                            amt = f' — <span class="badge badge-green">{pt.amount}</span>' if pt.amount else ''
                            st.markdown(f'• {pt.description}{amt}', unsafe_allow_html=True)
                    else:
                        st.caption("No payment terms found")
                    st.markdown('</div>', unsafe_allow_html=True)

                    if analysis.jurisdiction:
                        st.markdown(
                            f'<div class="card"><div class="card-title">🌍 Jurisdiction</div>{analysis.jurisdiction}</div>',
                            unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="card-title">⚠️ Risks & Liabilities</div>', unsafe_allow_html=True)
                    if analysis.risks:
                        for r in analysis.risks:
                            sev = r.severity.lower() if r.severity else "low"
                            css = f"risk-{sev}" if sev in ["high","medium","low"] else "risk-low"
                            bc = "badge-red" if sev=="high" else ("badge-yellow" if sev=="medium" else "badge-green")
                            st.markdown(
                                f'<div class="{css}"><span class="badge {bc}">{r.severity}</span> &nbsp; {r.description}</div>',
                                unsafe_allow_html=True)
                    else:
                        st.caption("No risks identified")

                    st.markdown('<br>', unsafe_allow_html=True)
                    st.markdown('<div class="card"><div class="card-title">📌 Obligations</div>', unsafe_allow_html=True)
                    for ob in analysis.obligations:
                        st.markdown(f"• {ob}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown('<div class="card"><div class="card-title">🚪 Termination Conditions</div>', unsafe_allow_html=True)
                    if analysis.termination_conditions:
                        for tc in analysis.termination_conditions:
                            st.markdown(f"• {tc}")
                    else:
                        st.caption("No termination conditions found")
                    st.markdown('</div>', unsafe_allow_html=True)

                    if analysis.penalty_clauses:
                        st.markdown('<div class="card"><div class="card-title">⚡ Penalty Clauses</div>', unsafe_allow_html=True)
                        for pc in analysis.penalty_clauses:
                            st.markdown(f"• {pc}")
                        st.markdown('</div>', unsafe_allow_html=True)

    # ── TAB 2 ─────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("### 💬 Ask Anything About the Contract")

        PRESET_QUERIES = [
            "Who are the parties involved?",
            "What are the key obligations?",
            "List all risks and liabilities",
            "What are the payment terms?",
            "What are the termination conditions?",
            "Are there any penalty clauses?",
            "What jurisdiction governs this?",
            "What are the important deadlines?",
        ]

        col_q, col_p = st.columns([2, 1])

        with col_p:
            st.markdown("**⚡ Quick Questions**")
            for q in PRESET_QUERIES:
                if st.button(q, use_container_width=True, key=f"preset_{q}"):
                    st.session_state["qa_query"] = q

        with col_q:
            query = st.text_area(
                "Your question:",
                value=st.session_state.get("qa_query", ""),
                height=120,
                placeholder="e.g. What happens if a party breaches the contract?"
            )

            if st.button("🔍 Get Structured Answer", type="primary") and query:
                with st.spinner("⚡ Analyzing + parsing structured output..."):
                    structured, raw, sources, error = ask_question(
                        st.session_state["vectorstore"], query, st.session_state["model"]
                    )
                    st.session_state["qa_result"] = structured
                    st.session_state["qa_raw"] = raw
                    st.session_state["qa_error"] = error

            if "qa_result" in st.session_state:
                qa = st.session_state["qa_result"]
                error = st.session_state["qa_error"]

                if error:
                    st.error(f"Parsing error: {error}")
                    st.text(st.session_state["qa_raw"])
                elif qa:
                    st.markdown(
                        f'<div class="summary-box">💬 <strong>Answer</strong><br><br>{qa.answer}</div>',
                        unsafe_allow_html=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown('<div class="card"><div class="card-title">📄 Relevant Clauses</div>', unsafe_allow_html=True)
                        for clause in qa.relevant_clauses:
                            st.markdown(f'<span class="tag">{clause}</span>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with c2:
                        conf = qa.confidence.lower() if qa.confidence else "medium"
                        conf_css = f"confidence-{conf}" if conf in ["high","medium","low"] else "confidence-medium"
                        st.markdown(
                            f'<div class="card"><div class="card-title">🎯 Confidence</div>'
                            f'<span class="{conf_css}">{qa.confidence}</span></div>',
                            unsafe_allow_html=True)

                    st.markdown('<div class="card"><div class="card-title">💡 Follow-up Suggestions</div>', unsafe_allow_html=True)
                    for s in qa.follow_up_suggestions:
                        if st.button(f"→ {s}", key=f"followup_{s}"):
                            st.session_state["qa_query"] = s
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

    # ── TAB 3 ─────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("### 🗂️ Raw JSON Schema Output")
        st.caption("Exact JSON returned by LLM — validated against Pydantic schema")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Full Analysis JSON")
            if "full_analysis" in st.session_state and st.session_state["full_analysis"]:
                json_str = json.dumps(st.session_state["full_analysis"].model_dump(), indent=2)
                st.markdown(f'<div class="json-box"><pre>{json_str}</pre></div>', unsafe_allow_html=True)
                st.download_button("⬇️ Download JSON", data=json_str,
                                   file_name="contract_analysis.json", mime="application/json")
            else:
                st.info("Run Full Analysis first")

        with c2:
            st.markdown("#### Q&A JSON")
            if "qa_result" in st.session_state and st.session_state["qa_result"]:
                json_str = json.dumps(st.session_state["qa_result"].model_dump(), indent=2)
                st.markdown(f'<div class="json-box"><pre>{json_str}</pre></div>', unsafe_allow_html=True)
                st.download_button("⬇️ Download JSON", data=json_str,
                                   file_name="qa_result.json", mime="application/json")
            else:
                st.info("Ask a question first")

else:
    st.markdown("""
    <div style="text-align:center; padding:3rem; color:#9ca3af;">
        <div style="font-size:4rem">⚖️</div>
        <h3 style="font-family:'DM Serif Display',serif; color:#1a1a2e;">Upload a contract to get started</h3>
        <p>Add your Groq API key in the sidebar, then upload a PDF or TXT contract above.</p>
    </div>
    """, unsafe_allow_html=True)
