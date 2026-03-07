from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(file_path: str):
    #Load the documnets and split it into chunks

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    else:
        loader = TextLoader(file_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators = ["\n\n", "\n", " ", ""]

    )

    chunks = splitter.split_documents(documents)
    print(f"✅ Loaded {len(documents)} pages → {len(chunks)} chunks")
    return chunks