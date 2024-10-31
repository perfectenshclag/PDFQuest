import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import fitz  # PyMuPDF for reading PDF text directly
import io
import time

# Load environment variables for API keys
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Initialize embeddings and language model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.2-90b-text-preview",max_tokens=4048)
llm.temperature = 0.45



# Prompt setup for generating question papers
prompt_template = """
    You are an expert who writes UPSC and other government exams paper answers based on the query and the provided context.
    Context: {context}
    
    Instructions:
    - Answer based on the context only. Do not add information that is not related to query and context
    - Answer in format like an expert UPSC teacher
    - Provide a detailed and precise explanation as if you are answering as a candidate.
    - List associated topics related to the answer.
    - Summarize the answer at the end.
    - For each part of the answer, specify the source name, page number, and reference details from the context in the end

    Question: {input}
    """
prompt = ChatPromptTemplate.from_template(prompt_template, kwargs=45)

# Define PDF generation function with improved styling
def create_pdf(answer_content):
    """Generate a styled PDF document using reportlab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle', parent=styles['Title'], fontSize=22, spaceAfter=12, textColor=colors.HexColor("#003366"), alignment=1)
    question_style = ParagraphStyle(
        'QuestionStyle', parent=styles['Heading2'], fontSize=16, spaceAfter=10, textColor=colors.green,)
    answer_style = ParagraphStyle(
        'AnswerStyle', parent=styles['BodyText'], fontSize=14, spaceAfter=8, leftIndent=10, rightIndent=10,)
    summary_style = ParagraphStyle(
        'SummaryStyle', parent=styles['BodyText'], fontSize=14, spaceAfter=15, fontName="Helvetica-Bold", textColor=colors.blue)

    # Create elements list for PDF
    elements = []

    # Add Title
    title = "📄 Generated Question Paper Answer"
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Add Question
    question_text = f"<b>Question:</b> {user_prompt}"
    elements.append(Paragraph(question_text, question_style))
    elements.append(Spacer(1, 0.1 * inch))

    # Add Answer Content
    answer_lines = answer_content.split("\n\n")
    for line in answer_lines:
        elements.append(Paragraph(line, answer_style))
        elements.append(Spacer(1, 0.1 * inch))
    
    # Add Summary
    summary_text = "📌 Summary:\n"
    elements.append(Paragraph(summary_text, summary_style))

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Document upload and embedding creation
def convert_pdf_to_text(pdf_bytes):
    """Extract text directly from each page of a PDF."""
    text = ""
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")  # Open PDF from bytes
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += f"\n\nPage {page_num + 1}:\n" + page.get_text("text")
    pdf_document.close()
    return text

# Updated function to process uploaded PDFs with filename metadata
def create_vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = embeddings
        docs = []  # Initialize docs list here
        with st.spinner("🔄 Processing PDFs and creating embeddings..."):
            for uploaded_file in uploaded_files:
                pdf_text = convert_pdf_to_text(uploaded_file.read())
                doc = Document(
                    page_content=pdf_text, 
                    metadata={"source": uploaded_file.name}  # Store PDF name here
                )
                docs.append(doc)  # Now this will work correctly

        # Split and embed documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(docs)
        
        # Create vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


st.title("📚 RAG Application for Answer Retrieval")

# Upload multiple PDF files
# Upload multiple PDF files
uploaded_files = st.file_uploader("📁 Upload multiple PDFs", type="pdf", accept_multiple_files=True)

# Initialize the vector database when files are uploaded
if uploaded_files:
    create_vector_embedding(uploaded_files)
    st.button("Document Embedding")
    st.write("✅ Vector Database is ready!")

# User query input
user_prompt = st.text_input("🔍 Enter your query")

if user_prompt and "vectors" in st.session_state:
    # Chain setup for question-answering with retrieved context
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    with st.spinner("💡 Generating answer..."):
        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(f"⏱️ Response time: {time.process_time() - start:.2f} seconds")

    # Retrieve answer
    st.write("### 💡 Answer:")
    st.write(response['answer'])

    # PDF generation and download button
    pdf_buffer = create_pdf(response['answer'])
    st.download_button(
        label="💾 Download Answer as PDF",
        data=pdf_buffer,
        file_name="Answer.pdf",
        mime="application/pdf"
    )
