from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import easyocr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from googletrans import Translator
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
import shutil

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual PDF Translator & QA",
    description="Convert Indian language PDFs to English and ask questions",
    version="1.0.0"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

print(f"API Key loaded: {api_key[:10]}...")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)

# Initialize embeddings
hugging_api_key = os.getenv("HuggingFace")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/LaBSE",
    model_kwargs={"device": "cpu"}
)

# Initialize translator
translator = Translator()

# Upload directory
UPLOAD_DIRECTORY = "Document"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Language mapping
lang_map = {
    "Marathi": ['mr', 'en'],
    "Assamese": ['as', 'en'],
    "Urdu": ['ur', 'en'],
    "Telugu": ['te', 'en'],
    "Kannada": ['kn', 'en'],
    "Malayalam": ['ml', 'en'],
    "Odia": ['or', 'en'],
    "English": ['en']
}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Server is running"}


@app.post("/multilingual_chat_bot/")
async def multilingual_chat_bot(
    file: UploadFile = File(...),
    doc_lang: str = Form(...),
    Final_language: str = Form(...),
    input_prompt: str = Form(...)
):
    """
    Process PDF, translate from Indian language to English, and answer questions
    
    Parameters:
    - file: PDF file to upload
    - doc_lang: Source language (Marathi, Assamese, Urdu, Telugu, Kannada, Malayalam, Odia, English)
    - Final_language: Target language (English)
    - input_prompt: Question about the document
    """
    
    try:
        # Validate language
        if doc_lang not in lang_map:
            return {"error": f"Language '{doc_lang}' not supported. Supported: {list(lang_map.keys())}"}
        
        # Create file path
        file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
        
        # Save uploaded file
        with open(file_location, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"Processing file: {file.filename} in language: {doc_lang}")
        
        # Initialize OCR reader
        reader = easyocr.Reader(lang_map[doc_lang], download_enabled=True)
        
        # Load PDF
        documents = []
        loader = PyPDFLoader(file_location)
        docs = loader.load()
        
        # Process each page
        for i, page in enumerate(docs):
            page.metadata["source"] = file.filename
            page.metadata["path"] = file_location
            page.metadata["page"] = i
            
            # If page content is empty, use OCR
            if not page.page_content.strip():
                ocr_text = ""
                images = page.images if hasattr(page, "images") else []
                
                for img in images:
                    ocr_result = reader.readtext(img, detail=0)
                    ocr_text += " ".join(ocr_result) + "\n"
                
                # Translate OCR text to English
                if ocr_text.strip():
                    page.page_content = translator.translate(
                        ocr_text, 
                        src_lang=lang_map[doc_lang][0],
                        dest_lang='en'
                    ).text
            else:
                # Translate existing text to English
                if doc_lang != "English":
                    page.page_content = translator.translate(
                        page.page_content,
                        src_lang=lang_map[doc_lang][0],
                        dest_lang='en'
                    ).text
        
        documents.extend(docs)
        
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an assistant for document question-answering. "
             "The retrieved context is in English (translated from Indian languages). "
             "Answer the user's question based on the provided context. "
             "Keep the answer concise and accurate (max 5 sentences). "
             "If the answer is not in the context, say 'I cannot find this information in the document.'\n\n{context}"),
            ("human", "{input}")
        ])
        
        # Create RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # Get response
        response = rag_chain.invoke({"input": input_prompt})
        
        # Clean up uploaded file
        if os.path.exists(file_location):
            os.remove(file_location)
        
        return {
            "response": response['answer'],
            "full_response": response,
            "success": True
        }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "error": str(e),
            "success": False
        }


# Entrypoint for deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)




