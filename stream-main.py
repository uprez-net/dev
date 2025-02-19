import streamlit as st
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_parse import LlamaParse
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
import nest_asyncio
from llama_index.core import Settings
from fpdf import FPDF
import json
from pinecone import Pinecone
from dotenv import load_dotenv
import os
from PIL import Image
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import pytesseract
from pdf2image import convert_from_path
import glob
import PyPDF2
import pdfplumber
import re

# Load environment variables
load_dotenv()
LLAMA_KEY = os.getenv('LLAMA_KEY')
GOOGLE_API = os.getenv('GOOGLE_API')
PINECONE_KEY = os.getenv('PINECONE_API_KEY')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_KEY')

# Initialize Gemini and Pinecone
model_name = "models/embedding-001"
embed_model = GeminiEmbedding(model_name=model_name, api_key=GOOGLE_API, title="this is a document")
llm = Gemini(model="models/gemini-1.5-flash", api_key=GOOGLE_API)
Settings.embed_model = embed_model
Settings.llm = llm
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("rules-and-regulation")

# Function definitions
def convert_pdf_to_images(pdf_path, save_dir, poppler_path):
    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    if os.path.exists(save_dir):
        for file in os.listdir(save_dir):
            if file.endswith('.jpg'):
                os.remove(os.path.join(save_dir, file))
    else:
        os.makedirs(save_dir, exist_ok=True)
    for i in range(len(images)):
        images[i].save(os.path.join(save_dir, f'page_{i+1}.jpg'), 'JPEG')

def perform_ocr_on_images(image_dir):
    data = ""
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(image_dir, image_file)
            try:
                with Image.open(image_path) as img:  # Use a context manager to ensure the file is closed
                    text = pytesseract.image_to_string(img)
                    data += text
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
    
    with open("output.txt", "w") as file:
        file.write(data)
    print("OCR Process completed")

def process_ocr(filepath):
    raw_documents = TextLoader(filepath).load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db = FAISS.from_documents(documents, OpenAIEmbeddings())
    data = {}
    queries = {
        'Balance sheet': "Extract all information regarding Balance Sheet",
        'Profit and Loss': "Extract all information regarding Profit and loss statement",
        'Cash flow statement': "Extract all information regarding Cash flow statement",
        'Notes to account': "Extract all information regarding Notes to account"
    }
    for key, query in queries.items():
        docs = db.similarity_search(query)
        data[key] = docs[0].page_content
    return data

def generate_txt_from_OCR(ocr_data):
    json_content = json.dumps(ocr_data, indent=4)
    with open("ocr.txt", "w") as file:
        file.write(json_content)

def parse_ocr_to_markdown(directory_path):
    parsed_data = []
    parser = LlamaParse(api_key=LLAMA_KEY, result_type="markdown",
                        content_guideline_instruction="This document is a financial report with charts, ratios, balance sheets, and other key financial data. Be precise in answering questions.")
    file_name = "ocr.txt"
    file_path = os.path.join(directory_path, file_name)
    if os.path.exists(file_path):
        try:
            parsed_data += parser.load_data(file_path)
        except Exception as e:
            st.error(f"Error parsing {file_name}: {e}")
    else:
        st.error(f"File {file_name} not found in {directory_path}")
    return parsed_data

def initialize_vector_store(parsed_data):
    reranker = FlagEmbeddingReranker(top_n=5, model="BAAI/bge-reranker-large")
    raw_index = VectorStoreIndex.from_documents(documents=parsed_data, embed_model=embed_model)
    return raw_index.as_query_engine(similarity_top_k=15, node_postprocessors=[reranker])

def retrieve_and_generate(query, raw_query_engine):
    query_embedding = embed_model.get_text_embedding(query)
    results = index.query(namespace="new-namespace", vector=query_embedding, top_k=4, include_values=False,
                          include_metadata=True)
    matches_result = results.matches[0].metadata.get("source_text")
    response = raw_query_engine.query(query)
    return results, response

def check_compliance(response, compliance_rules):
    prompt = f"Does the following response follow the compliance rules?\n\nResponse:\n{response}\n\nCompliance Rules:\n{compliance_rules}"
    return llm.complete(prompt=prompt)

# Streamlit UI
st.title("Financial Document Compliance Checker")

# File upload
uploaded_file = st.file_uploader("Upload a financial document (PDF)", type="pdf")
query = st.text_input("Enter your query to check for compliance")

if uploaded_file is not None and query:
    with st.spinner('Processing document...'):
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Convert PDF to images and perform OCR
        convert_pdf_to_images("temp.pdf", "images", 'C:\\Users\\satis\\Downloads\\poppler\\poppler-24.08.0\\Library\\bin')
        perform_ocr_on_images("images")
        
        # Process OCR data
        ocr_data = process_ocr("output.txt")
        generate_txt_from_OCR(ocr_data)
        
        # Parse OCR to Markdown
        parsed_data = parse_ocr_to_markdown(os.getcwd())
        st.write(f"Total parsed documents: {len(parsed_data)}")
        
        # Initialize vector store and reranker
        raw_query_engine = initialize_vector_store(parsed_data)
        
        # Retrieve and generate responses
        compliance_rules, res = retrieve_and_generate(query, raw_query_engine)
        st.write("Compliance Rules:")
        st.write(compliance_rules)
        st.write("Response:")
        st.write(res)
        
        # Check compliance
        compliance_check = check_compliance(res, compliance_rules)
        st.write("Compliance Check:")
        st.write(compliance_check)

if __name__ == "__main__":
    nest_asyncio.apply()