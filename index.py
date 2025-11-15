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
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.warning("GOOGLE_API_KEY not found in environment variables")

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key) if api_key else None
except Exception as e:
    logger.error(f"Error initializing LLM: {e}")
    llm = None

# Initialize embeddings
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/LaBSE",
        model_kwargs={"device": "cpu"}
    )
except Exception as e:
    logger.error(f"Error initializing embeddings: {e}")
    embeddings = None

# Initialize translator
translator = Translator()

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


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multilingual PDF Translator & QA API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "translate": "/multilingual_chat_bot/",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Server is running",
        "llm_ready": llm is not None,
        "embeddings_ready": embeddings is not None
    }


@app.post("/multilingual_chat_bot/")
async def multilingual_chat_bot(
    file: UploadFile = File(...),
    doc_lang: str = Form(...),
    final_language: str = Form(...),
    input_prompt: str = Form(...)
):
    """
    Process PDF, translate from Indian language to English, and answer questions
    
    Parameters:
    - file: PDF file to upload
    - doc_lang: Source language (Marathi, Assamese, Urdu, Telugu, Kannada, Malayalam, Odia, English)
    - final_language: Target language (English)
    - input_prompt: Question about the document
    """
    
    try:
        # Check if LLM and embeddings are initialized
        if not llm or not embeddings:
            return {
                "error": "System not properly initialized",
                "success": False
            }
        
        # Validate language
        if doc_lang not in lang_map:
            return {
                "error": f"Language '{doc_lang}' not supported. Supported: {list(lang_map.keys())}",
                "success": False
            }
        
        # Validate file
        if not file.filename.endswith('.pdf'):
            return {
                "error": "Only PDF files are supported",
                "success": False
            }
        
        logger.info(f"Processing file: {file.filename} in language: {doc_lang}")
        
        # Use temporary directory for file storage (Vercel compatible)
        with tempfile.TemporaryDirectory() as temp_dir:
            file_location = os.path.join(temp_dir, file.filename)
            
            # Save uploaded file
            with open(file_location, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Initialize OCR reader
            reader = easyocr.Reader(lang_map[doc_lang], download_enabled=True)
            
            # Load PDF
            documents = []
            loader = PyPDFLoader(file_location)
            docs = loader.load()
            
            logger.info(f"Loaded {len(docs)} pages from PDF")
            
            # Process each page
            for i, page in enumerate(docs):
                page.metadata["source"] = file.filename
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
                        try:
                            page.page_content = translator.translate(
                                ocr_text,
                                src_lang=lang_map[doc_lang][0],
                                dest_lang='en'
                            ).text
                        except Exception as e:
                            logger.warning(f"Translation error on page {i}: {e}")
                            page.page_content = ocr_text
                else:
                    # Translate existing text to English if not already in English
                    if doc_lang != "English":
                        try:
                            page.page_content = translator.translate(
                                page.page_content,
                                src_lang=lang_map[doc_lang][0],
                                dest_lang='en'
                            ).text
                        except Exception as e:
                            logger.warning(f"Translation error on page {i}: {e}")
            
            documents.extend(docs)
            
            # Split documents into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            splits = splitter.split_documents(documents)
            
            logger.info(f"Created {len(splits)} document chunks")
            
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
            
            logger.info(f"Processing question: {input_prompt}")
            
            # Get response
            response = rag_chain.invoke({"input": input_prompt})
            
            logger.info("Response generated successfully")
            
            return {
                "response": response['answer'],
                "success": True,
                "file_name": file.filename,
                "source_language": doc_lang,
                "target_language": final_language
            }
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "success": False
        }
