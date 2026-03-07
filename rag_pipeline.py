import os
import json
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ingest import load_and_chunk
from schemas import ContractAnalysis, QuestionAnswer

load_dotenv()


# ── 1. Prompt Templates ───────────────────────────────────────────────────────

FULL_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert legal AI assistant. Analyze the legal contract and extract ALL structured information.

Contract Text:
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

JSON:"""
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

JSON:"""
)


# ── 2. HuggingFace Embeddings ─────────────────────────────────────────────────

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ── 3. Groq LLM ───────────────────────────────────────────────────────────────

def get_llm(model_name: str = "llama3-70b-8192"):
    return ChatGroq(
        model=model_name,
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )


# ── 4. Vector Store ───────────────────────────────────────────────────────────

def build_vectorstore(file_path: str):
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


# ── 5. LCEL Chain Builder ─────────────────────────────────────────────────────

def build_chain(vectorstore, prompt, llm):
    """Modern LCEL chain — no deprecated RetrievalQA"""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── 6. Structured Output Parsing ──────────────────────────────────────────────

def parse_structured_output(raw_text: str, schema_class):
    try:
        clean = raw_text.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0].strip()
        elif "```" in clean:
            clean = clean.split("```")[1].split("```")[0].strip()

        data = json.loads(clean)
        validated = schema_class(**data)
        return validated, None

    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {str(e)}"
    except Exception as e:
        return None, f"Schema validation error: {str(e)}"


# ── 7. Full Contract Analysis ─────────────────────────────────────────────────

def full_contract_analysis(vectorstore, model_name: str = "llama3-8b-8192"):
    llm = get_llm(model_name)
    chain, retriever = build_chain(vectorstore, FULL_ANALYSIS_PROMPT, llm)

    query = "Perform a complete structured analysis of this legal contract."
    raw_output = chain.invoke(query)
    sources = retriever.invoke(query)

    structured, error = parse_structured_output(raw_output, ContractAnalysis)
    return structured, raw_output, sources, error


# ── 8. Custom Question ────────────────────────────────────────────────────────

def ask_question(vectorstore, question: str, model_name: str = "llama3-8b-8192"):
    llm = get_llm(model_name)
    chain, retriever = build_chain(vectorstore, QA_PROMPT, llm)

    raw_output = chain.invoke(question)
    sources = retriever.invoke(question)

    structured, error = parse_structured_output(raw_output, QuestionAnswer)
    return structured, raw_output, sources, error
