#unsrt-OCR-streamlit
import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import os
import json
from llama_index.core import Document  # Import the Document class
from llama_parse import LlamaParse
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
import nest_asyncio
from llama_index.core import Settings
from fpdf import FPDF
from pinecone import Pinecone
from dotenv import load_dotenv
import tempfile
from PIL import Image
import re

# Load environment variables
load_dotenv()
LLAMA_KEY = os.getenv('LLAMA_KEY')
GOOGLE_API = os.getenv('GOOGLE_API')
PINECONE_KEY = os.getenv('PINECONE_API_KEY')

# Initialize Gemini and Pinecone
model_name = "models/embedding-001"
embed_model = GeminiEmbedding(model_name=model_name, api_key=GOOGLE_API, title="this is a document")
llm = Gemini(model="models/gemini-1.5-flash", api_key=GOOGLE_API)
Settings.embed_model = embed_model
Settings.llm = llm
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index("rules-and-regulation")

# Function to perform OCR and chunking
def ocr_and_chunking(uploaded_file):
    # Save the uploaded file to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Convert PDF to images with reduced DPI to save memory
        images = convert_from_path(pdf_path, poppler_path='C:\\Users\\satis\\Downloads\\poppler\\poppler-24.08.0\\Library\\bin', dpi=200)

        # Perform OCR on each image
        combined_text = ""
        for i, image in enumerate(images):
            try:
                # Convert image to grayscale to improve OCR accuracy
                image = image.convert("L")
                text = pytesseract.image_to_string(image)
                combined_text += text + "\n"
            except Exception as e:
                st.error(f"Error processing page {i + 1}: {e}")
                continue

        # Chunk the combined text
        chunks = chunk_financial_document(combined_text)

        # Save chunks to a JSON file
        with open('chunks.json', 'w') as json_file:
            json.dump(chunks, json_file, indent=4)

        return chunks

# Function to chunk the financial document
def chunk_financial_document(extracted_text):
    sections = {
        "Consolidated Balance Sheet": r"Consolidated Balance Sheet.*?(?=\n\s*\n)",
        "Consolidated Profit & Loss Statement": r"Consolidated Profit & Loss Statement.*?(?=\n\s*\n)",
        "Consolidated Cash Flow Statement": r"Consolidated Cash Flow Statement.*?(?=\n\s*\n)",
        "Financial Liabilities": r"Financial Liabilities.*?(?=\n\s*\n)",
    }

    chunks = {}
    for section_name, pattern in sections.items():
        match = re.search(pattern, extracted_text, re.DOTALL | re.IGNORECASE)
        if match:
            start_index = match.start()
            end_index = start_index + len(match.group(0))
            next_lines = extracted_text[end_index:].split('\n')[:100]
            chunks[section_name] = match.group(0).strip() + '\n' + '\n'.join(next_lines)
        else:
            fallback_pattern = r"(?i)" + re.escape(section_name) + r".*?(?=\n\n)"
            fallback_match = re.search(fallback_pattern, extracted_text, re.DOTALL)
            if fallback_match:
                start_index = fallback_match.start()
                end_index = start_index + len(fallback_match.group(0))
                next_lines = extracted_text[end_index:].split('\n')[:100]
                chunks[section_name] = fallback_match.group(0).strip() + '\n' + '\n'.join(next_lines)
            else:
                chunks[section_name] = "Section not found"

    return chunks

# Function to retrieve and generate responses
def retrieve_and_generate(query, chunks):
    # Convert chunks dictionary into a list of Document objects
    documents = []
    for section_name, text in chunks.items():
        if text != "Section not found":  # Skip invalid sections
            doc = Document(
                text=text,
                metadata={"section_name": section_name}
            )
            documents.append(doc)

    # Initialize the reranker
    reranker = FlagEmbeddingReranker(top_n=5, model="BAAI/bge-reranker-large")

    # Create the VectorStoreIndex from the documents
    raw_index = VectorStoreIndex.from_documents(documents=documents, embed_model=embed_model)

    # Create a query engine
    raw_query_engine = raw_index.as_query_engine(similarity_top_k=15, node_postprocessors=[reranker])

    # Perform semantic search on Pinecone
    query_embedding = embed_model.get_text_embedding(query)
    results = index.query(namespace="new-namespace", vector=query_embedding, top_k=4, include_values=False, include_metadata=True)
    matches_result = results.matches[0].metadata.get("source_text")

    # Generate a response using the query engine
    response = raw_query_engine.query(query)
    return results, response

# Function to check compliance
def check_compliance(response, compliance_rules):
    prompt = f"Does the following response follow the compliance rules?\n\nResponse:\n{response}\n\nCompliance Rules:\n{compliance_rules}"
    return llm.complete(prompt=prompt)

# Streamlit App
def main():
    st.title("Financial Document Compliance Checker")
    st.write("Upload a financial document (PDF) to check compliance.")

    # Step 1: Upload Financial Document
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        st.write("File uploaded successfully!")

        # Step 2: Perform OCR + Chunking
        st.write("Performing OCR and chunking...")
        chunks = ocr_and_chunking(uploaded_file)
        st.write("OCR and chunking completed!")

        # Step 3: Query Input
        query = st.text_input("Enter your query (e.g., 'cash flow statement'):")
        if query:
            st.write(f"Query: {query}")

            # Step 4: RAG + Compliance Check
            st.write("Retrieving and generating responses...")
            compliance_rules, response = retrieve_and_generate(query, chunks)
            st.write("Compliance rules retrieved:")
            st.write(compliance_rules)
            st.write("Response generated:")
            st.write(response)

            # Step 5: Compliance Check
            st.write("Checking compliance...")
            compliance_check = check_compliance(response, compliance_rules)
            st.write("Compliance check result:")
            st.write(compliance_check)

if __name__ == "__main__":
    main()
