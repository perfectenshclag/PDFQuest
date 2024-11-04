
# 📚 pdfquest - Your Study Buddy! 🚀

**pdfquest** is here to make studying smarter, not harder! Are you a student prepping for exams or a researcher deep-diving into stacks of PDFs? With **pdfquest**, you can focus on *just* the information you need, getting answers straight from your own notes, research papers, or study guides. No extra fluff, just the facts, right from your sources! 🌟

## 🎉 What is pdfquest?

**pdfquest** is a question-answering app powered by **Retrieval-Augmented Generation (RAG)**. It digs through your PDF documents, pulls out only the relevant sections, and gives you concise, context-based answers. It’s like a search engine, but for your own study material—perfect for tackling exam prep and research projects!

## 🔍 Why pdfquest?

- **Streamlined Study**: Skip the skimming! Ask a question, and pdfquest fetches the answer from your uploaded files.
- **Context Matters**: Only the context you provide matters, so your answers are authentic and sourced.
- **Exam Hero**: Great for focusing on the most relevant material, especially when time is ticking!

## 🚀 Key Features

- **RAG-powered Q&A**: Ask questions and get answers directly from your documents.
- **Fast Vector Search**: Uses FAISS embeddings for fast, accurate search results.
- **Document Splitting**: Breaks down large PDFs into manageable chunks so nothing is missed.
- **Interactive UI**: Simple Streamlit interface—type in your question and go!
- **Accurate Responses**: Uses only the provided context, ensuring answers come from your source material.

## 🛠️ How It Works

1. **Load Documents**: Import your PDFs (e.g., research papers, lecture notes).
2. **Split and Embed**: Each document is split and embedded into a searchable vector space.
3. **Ask Away**: Type in a question, and pdfquest retrieves relevant document parts to answer you, with no outside info mixed in.

![RAG Workflow Diagram](https://storage.googleapis.com/lightning-avatars/litpages/01hmw7z4vjb2tpfdyz811061zm/ff7a557f-6a43-4bcc-8b0d-b8e8a5943fca.png)

## 🔧 Built With

- **Streamlit**: The app’s front-end, making it simple and interactive.
- **LangChain**: Framework for handling document retrieval and RAG.
- **FAISS**: Fast vector database for accurate content search.
- **Groq AI and HuggingFace**: For AI-powered processing and embedding creation.

## 🏁 Get Started

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/yourusername/pdfquest.git
   cd pdfquest
   ```

2. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Keys**:
   In the root folder, create a `.env` file and add your API keys:
   ```plaintext
   OPENAI_API_KEY="your_openai_key"
   GROQ_API_KEY="your_groq_key"
   HF_TOKEN="your_huggingface_token"
   ```

4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

5. **Ask Your First Question**:
   - Upload PDFs and ask questions—pdfquest will pull the best answer from your content.

## 🧑‍🎓 Example Use Case

Studying for an exam? Instead of reading *everything*, upload your lecture notes and research papers to **pdfquest**. Just ask specific questions, like “What is the process of photosynthesis?” or “Summarize the main points of Chapter 3,” and get precise answers. 💥

## 📜 Sample Code Snippet

```python
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# A sample prompt for Q&A
prompt = ChatPromptTemplate.from_template(
    """
    Answer based on the provided context.
    <context>
    {context}
    <context>
    Question: {input}
    """
)
```

## 🤝 Contributions

Ideas to improve pdfquest? Feel free to fork, tweak, and submit a pull request! Join us in making studying easier for everyone!

---

Enjoy smarter study sessions with **pdfquest**! 🧠💬
