# ⚖️ Legal AI Assistant

A domain-specific AI assistant that analyzes legal contracts and extracts structured insights like clauses, risks, parties, and obligations — built using RAG (Retrieval Augmented Generation) with LangChain.

---

## 🚀 Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core language |
| **LLM APIs** | Groq LLaMA3 — free & blazing fast |
| **Prompt Templates** | LangChain `PromptTemplate` for domain-specific legal prompts |
| **Structured Output Parsing** | `parse_structured_output()` — strips fences → `json.loads()` → Pydantic validation |
| **JSON Schema Generation** | Pydantic `BaseModel` — `ContractAnalysis` & `QuestionAnswer` schemas |
| **HuggingFace Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` — runs locally, FREE |
| **FAISS** | Local vector store for fast similarity search |
| **Streamlit** | Web UI |

---

## 📁 Project Structure

```
legal_ai_assistant/
│
├── app.py                  ← Streamlit UI (run this)
├── rag_pipeline.py         ← HuggingFace + Groq + RAG logic
├── schemas.py              ← Pydantic models / JSON schema
├── ingest.py               ← Document loading & chunking
├── requirements.txt        ← All dependencies
├── .env                    ← Your Groq API key (never push this)
├── .gitignore              ← Ignores .env, venv, faiss index
│
└── faiss_legal_index/      ← Auto-created when you upload a contract
    ├── index.faiss
    └── index.pkl
```

---

## ⚙️ How It Works

```
Contract (PDF/TXT)
      ↓
  [ingest.py]        → RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
      ↓
  [HuggingFace]      → all-MiniLM-L6-v2 embeddings (runs locally)
      ↓
  [FAISS]            → Vector store saved to disk
      ↓
User Query → Retriever → Top 4 relevant chunks
                               ↓
                     [PromptTemplate]    ← domain-specific legal prompt
                               ↓
                     [Groq LLaMA3]      ← LLM API call
                               ↓
                  [parse_structured_output()]  ← strips markdown, json.loads()
                               ↓
                  [Pydantic Schema Validation] ← ContractAnalysis / QuestionAnswer
                               ↓
                     Structured JSON Output
```

---

## 🛠️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/legal-ai-assistant.git
cd legal-ai-assistant
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get your free Groq API key
- Go to [console.groq.com](https://console.groq.com)
- Sign up → API Keys → Create API Key

### 5. Create `.env` file
```
GROQ_API_KEY=your_groq_api_key_here
```

### 6. Run the app
```bash
streamlit run app.py
```

Open browser at **`http://localhost:8501`**

---

## 📦 Dependencies

```
langchain==0.3.0
langchain-community==0.3.0
langchain-huggingface==0.1.0
langchain-groq==0.2.0
faiss-cpu==1.8.0
streamlit==1.38.0
pypdf==4.3.1
python-dotenv==1.0.1
sentence-transformers==3.1.1
pydantic==2.9.0
```

---

## 🖥️ Features

### 🔍 Full Contract Analysis
Runs a complete structured analysis and extracts:
- 📝 Contract summary
- 👥 Parties involved (with roles)
- ⚠️ Risks & liabilities (with severity: High / Medium / Low)
- 📅 Key dates & deadlines
- 💰 Payment terms
- 📌 Obligations
- 🚪 Termination conditions
- ⚡ Penalty clauses
- 🌍 Jurisdiction

### 💬 Ask a Question
Ask anything about the contract and get back:
- Direct answer
- Relevant clauses
- Confidence level (High / Medium / Low)
- Follow-up question suggestions

### 🗂️ Raw JSON Output
- View the exact structured JSON returned by the LLM
- Download as `.json` file

---

## 📐 JSON Schema (Pydantic)

```python
class ContractAnalysis(BaseModel):
    summary: str
    parties: List[Party]
    risks: List[Risk]
    key_dates: List[KeyDate]
    payment_terms: List[PaymentTerm]
    obligations: List[str]
    termination_conditions: List[str]
    jurisdiction: Optional[str]
    penalty_clauses: List[str]

class QuestionAnswer(BaseModel):
    answer: str
    relevant_clauses: List[str]
    confidence: str
    follow_up_suggestions: List[str]
```

---

## 🔄 Daily Git Workflow

```bash
# Start of day
git pull origin main

# After coding
git add .
git commit -m "your message"
git push origin main
```

---

## ⚠️ .gitignore

Make sure these are never pushed to GitHub:
```
.env
venv/
faiss_legal_index/
__pycache__/
*.pyc
```

---

## 💰 Cost

| Component | Cost |
|---|---|
| HuggingFace Embeddings | ✅ FREE (runs locally) |
| Groq LLaMA3 API | ✅ FREE |
| FAISS Vector Store | ✅ FREE (runs locally) |
| **Total** | **$0** |

---

## 📚 Key Concepts Learned

- **Domain adaptation** — legal-specific prompt engineering
- **Information extraction** — clause and risk extraction
- **Structured generation** — forcing LLM to output valid JSON
- **Prompt engineering** — role + format + schema in prompt
- **RAG pipeline** — retriever + prompt + LLM chained together
- **LCEL** — LangChain Expression Language (modern chain syntax)

---

## 🙏 Credits

Built following the RAG with LangChain concepts 
