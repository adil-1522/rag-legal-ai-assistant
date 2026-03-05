import os
import json
from dotenv import load_dotenv

# ── Tech Stack ────────────────────────────────────────────────────────────────
# LLM API       → Groq (LLaMA3)
# Embeddings    → HuggingFace (local, free)
# Vector Store  → FAISS
# Prompt Template → LangChain PromptTemplate
# Structured Output Parsing → parse_structured_output()
# JSON Schema Generation    → schemas.py (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.chains import RetrievalQAWithSourcesChain
from langchain_core.prompts import PromptTemplate

from ingest import load_and_chunk
from schemas import ContractAnalysis, QuestionAnswer

load_dotenv()

#1.Prompt Template

FULL_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["context","question"],
    template="""
You are an expert legal AI assistant. Analyze the legal contract and extract all the structured information

Context Text:
{context}

{question}
You MUST respond with ONLY a valid JSON object.
No explanation, no markdown code fences, no extra text — just raw JSON.

Return this exact structure:
{{
  "summary": "2-3 sentence summary",
  "parties": [{{"name": "...", "role": "..."}}],
  "risks": [{{"description": "...", "severity": "High/Medium/Low"}}],
  "key_dates": [{{"event": "...", "date": "..."}}],
  "payment_terms": [{{"description": "...", "amount": "... or null"}}],
  "obligations": ["..."],
  "termination_conditions": ["..."],
  "jurisdiction": "... or null",
  "penalty_clauses": ["..."]
}}

JSON:

"""
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert legal AI assistant. Answer the question using only the contract provided.

Contract Context:
{context}

Question: {question}

You MUST respond with ONLY a valid JSON object.
No explanation, no markdown code fences, no extra text — just raw JSON.

Return this exact structure:
{{
  "answer": "detailed answer here",
  "relevant_clauses": ["clause 1", "clause 2"],
  "confidence": "High/Medium/Low",
  "follow_up_suggestions": ["question 1", "question 2", "question 3"]
}}

If not found in contract, set answer to: "This information is not in the provided contract."

JSON:
"""
)

#2. Hugging face embeddings (local,free)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings":True}
    )

#3. Groq LLM

def get_llm(model_name: str = "llama3-8b-8192"):
    return ChatGroq(
        model=model_name,
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )


#4. FAISS Vector Store
def build_vectorstore(file_path: str):
    """Chunk → Embed → Store in FAISS"""
    chunks = load_and_chunk(file_path)
    print("⏳ Generating HuggingFace embeddings locally...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_legal_index")
    print("✅ Vector store ready!")
    return vectorstore

def load_vectorstore():
    embeddings = get_embeddings()
    return FAISS.load_local(
        "faiss_legal_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
