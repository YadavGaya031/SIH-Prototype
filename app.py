from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_groq import ChatGroq
# from langchain_cohere import CohereEmebeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from dotenv import load_dotenv
import os
import io
import cohere
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
import tempfile

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_DIR = "vectorstore"

if not COHERE_API_KEY:
    raise ValueError("Missing COHERE_API_KEY in environment variables")

app = FastAPI()

llm = ChatGroq(
    api_key = GROQ_API_KEY,
    model="groq/compound-mini",
    temperature=0
)

class CohereEmbeddings(Embeddings):
    def __init__(self, api_key, model="embed-english-v3.0"):
        self.client = cohere.Client(api_key)
        self.model = model

    def embed_documents(self, texts):
        response = self.client.embed(texts=texts, model=self.model, input_type="search_document")
        return response.embeddings

    def embed_query(self, text):
        response = self.client.embed(texts=[text], model=self.model, input_type="search_query")
        return response.embeddings[0]

embeddings = CohereEmbeddings(api_key=COHERE_API_KEY)

try:
    vectordb = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
except Exception:
    vectordb = None

text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)

async def extract_text(file: UploadFile) -> str:
    ext = file.filename.lower().split('.')[-1]
    content = await file.read()

    if ext == "txt":
        return content.decode('utf-8')

    elif ext in ["pdf", "docx"]:
       
        suffix = f".{ext}"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(content)
            tmp.close()  

            if ext == "pdf":
                loader = PyPDFLoader(tmp.name)
            else:
                loader = Docx2txtLoader(tmp.name)

            docs = loader.load()
            return "\n".join([doc.page_content for doc in docs])
        finally:
            os.unlink(tmp.name) 

    elif ext == "html":
        soup = BeautifulSoup(content, 'html.parser')
        return soup.get_text()

    elif ext in ["png", "jpg", "jpeg", "tiff"]:
        image = Image.open(io.BytesIO(content))
        return pytesseract.image_to_string(image)

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vectordb
    try:
        text = await extract_text(file)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Failed to extract text from document.")

        doc = Document(page_content=text, metadata={"source": file.filename})

        chunks = text_splitter.split_documents([doc])
        chunk_texts = [chunk.page_content for chunk in chunks]
        chunk_embeddings = embeddings.embed_documents(chunk_texts)

        if vectordb is None:
            vectordb = FAISS.from_documents(chunks, embeddings)
        else:
            vectordb.add_documents(chunks)

        vectordb.save_local(DB_DIR)

        return {
            "message": f"Uploaded and ingested document '{file.filename}' successfully.",
            "num_chunks": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_documents(query: str = Body(..., embed=True)):
    try:
        if vectordb is None:
            return {"answer": "No documents ingested yet.", "source_doc_id": None}

        results = vectordb.similarity_search_with_score(query, k=3)

        if not results or len(results) == 0:
            return {"answer": "Sorry, I couldn't find relevant information.", "source_doc_id": None}

        top_chunks = [doc.page_content for doc, score in results]
        metadatas = [doc.metadata for doc, score in results]
        source_doc_id = metadatas[0].get("source") if metadatas else None
        routing = metadatas[0].get("routing") if metadatas else None

        context = "\n\n".join(top_chunks)

        prompt = f""" You are a helpful assistant. Based on the following document excerpts, answer the user's question precisely. Document excerpts: {context} User question: {query} Please provide a detailed answer. If the information is not available in the excerpts, say "I don't know". Answer: """ 
        # print("Prompt sent to LLM:", prompt) 
        response = llm.invoke(prompt) 
        print("Raw LLM response:", response.content) 
        answer = response.content.strip() 
        return { 
            "answer": answer, 
            "source_doc_id": source_doc_id, 
            "routing": routing 
            }
       
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Upload documents via /upload and ask questions via /chat."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
