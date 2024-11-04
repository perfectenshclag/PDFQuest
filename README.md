
# 📄 pdfquest 🚀

**pdfquest** is here to save the day for students and researchers alike! Imagine you’ve got a stack of PDFs—endless pages of content, and you just need answers to *very specific* questions. Why waste time scrolling through everything? Let **pdfquest** handle the heavy lifting for you! With a powerful Retrieval-Augmented Generation (RAG) model, it quickly finds relevant content from your documents, giving you concise, context-based answers. No more guesswork, just facts from *your* material.

## 🎯 Project Overview

**pdfquest** is an AI-powered app built with Streamlit and LangChain that helps students and researchers extract answers from a vast repository of PDF content. By using FAISS embeddings, it efficiently retrieves and processes the information, ensuring your answers are both relevant and accurate, based solely on the provided context.

## 💡 Features

- **RAG-powered Q&A**: Get answers to your queries directly from your documents.
- **FAISS Vector Embeddings**: Superfast and precise retrieval for relevant content.
- **Chunked Document Processing**: Splits large PDFs into manageable pieces, so nothing is missed.
- **Streamlit UI**: Intuitive, interactive interface—just type and get answers.
- **Context-based Responses**: Authentic responses only from the provided documents.

## ⚙️ How It Works

1. **Data Ingestion**: Load research papers or any study material in PDF format.
2. **Document Splitting**: Chunk documents for efficient retrieval.
3. **Embedding Creation**: Generate embeddings with FAISS for fast similarity search.
4. **RAG Q&A**: Enter a question, and pdfquest searches the docs to provide a contextually relevant answer.

## 🛠️ Tech Stack

- **Streamlit**: Front-end for an interactive app experience.
- **LangChain**: Framework for handling embeddings, document splitting, and RAG processes.
- **FAISS**: Handles the vector database, making retrieval blazing fast.
- **Groq AI**: Harness the power of Groq’s AI models for state-of-the-art NLP.
- **HuggingFace**: Supports robust embedding generation.

## 🏁 Quick Start Guide

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/yourusername/pdfquest.git
   cd pdfquest
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add Your API Keys**:
   Create a `.env` file in the root folder and add your keys:
   ```plaintext
   OPENAI_API_KEY="your_openai_key"
   GROQ_API_KEY="your_groq_key"
   HF_TOKEN="your_huggingface_token"
   ```

4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

5. **Start Asking**:
   - Upload PDFs of research papers or study guides.
   - Type your question in the input box.
   - Hit **Enter**, and let **pdfquest** find the best answer!

## 🧪 Use Case: Ace Your Exam Prep!

Got a pile of lecture notes, research papers, or e-books to study? **pdfquest** can help! Ask anything directly related to your material and get answers that pull directly from your PDFs. It’s like having a personal tutor that only reads *your* books. 🏆

## 📖 Sample Code Snippet

```python
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Sample prompt for the Q&A system
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    <context>
    {context}
    <context>
    Question:{input}
    """
)
```

## 🌟 Contributing

Have ideas to make **pdfquest** even better? Contributions are welcome! Feel free to fork the repo, make improvements, and submit a pull request.

---

Enjoy streamlined studying with **pdfquest**—your AI-powered study buddy! 🧠💡
