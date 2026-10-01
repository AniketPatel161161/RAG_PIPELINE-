# 🤖📚 RAG Pipeline: Retrieval-Augmented Generation with LangChain, ChromaDB & OpenAI


A complete, beginner-friendly **Retrieval-Augmented Generation (RAG)** pipeline built from scratch in a Jupyter/Colab notebook. It loads your own documents (text files and PDFs), splits them into chunks, converts them into vector embeddings, stores them in a **ChromaDB** vector database, and then answers questions by retrieving the most relevant chunks and passing them to an **OpenAI LLM**.

---

## 📑 Table of Contents

- [🌟 What is RAG?](#-what-is-rag)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🧰 Tech Stack](#-tech-stack)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🚀 Step-by-Step Walkthrough](#-step-by-step-walkthrough)
- [💻 Usage Example](#-usage-example)
- [🔧 Configuration Options](#-configuration-options)
- [🔐 Security Notes](#-security-notes)
- [🐞 Known Issues & Fixes](#-known-issues--fixes)
- [🔮 Future Improvements](#-future-improvements)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 🌟 What is RAG?

Large Language Models are powerful, but they can be **outdated**, **hallucinate**, and know nothing about **your private data**. **Retrieval-Augmented Generation (RAG)** fixes this by:

1. 🔍 **Retrieving** relevant information from an external knowledge base
2. 🧠 **Augmenting** the LLM prompt with that information
3. ✍️ **Generating** an answer grounded in your actual documents

The result: more **accurate**, **up-to-date**, and **context-aware** responses.

---

## ✨ Features

- 📄 Loads **plain text** (`.txt`) and **PDF** files (single files or a whole folder)
- ✂️ Smart **recursive chunking** with configurable size and overlap
- 🧬 Free, local **sentence-transformer embeddings** (no API cost for embeddings)
- 💾 **Persistent** vector storage with ChromaDB (data survives restarts)
- 🎯 **Semantic search** with similarity scores, ranking, and score threshold filtering
- 🧠 **LLM integration** with OpenAI via LangChain
- 🧱 Clean, **modular, class-based** design (`EmbeddingManager`, `VectorStoreManager`, `RAGRetriever`)

---

## 🏗️ Architecture

The pipeline has two phases.

### 📥 Phase 1: Ingestion

```
 📂 Data (TXT / PDF)
        │
        ▼
 📄 Document Loading      (TextLoader / PyPDFLoader)
        │
        ▼
 ✂️ Chunking              (RecursiveCharacterTextSplitter)
        │
        ▼
 🧬 Embedding Generation  (SentenceTransformer: all-MiniLM-L6-v2)
        │
        ▼
 💾 Vector Store          (ChromaDB, persistent)
```

### 📤 Phase 2: Retrieval & Generation

```
 ❓ User Query
        │
        ▼
 🧬 Query Embedding
        │
        ▼
 🔎 Similarity Search     (top-k chunks from ChromaDB)
        │
        ▼
 🧩 Context Augmentation  (context + query → prompt)
        │
        ▼
 🤖 LLM (OpenAI)          → ✅ Final Answer
```

---

## 🧰 Tech Stack

| Component | Tool / Library | Purpose |
|---|---|---|
| 🦜 Framework | `langchain`, `langchain-core`, `langchain-community` | Document objects & loaders |
| 📄 PDF parsing | `pypdf`, `pymupdf` | Reading PDF pages |
| ✂️ Text splitting | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| 🧬 Embeddings | `sentence-transformers` | `all-MiniLM-L6-v2` (384 dimensions) |
| 💾 Vector DB | `chromadb` | Persistent vector storage & search |
| 📐 Similarity | `scikit-learn` | Cosine similarity utilities |
| 🤖 LLM | `langchain-openai` | `ChatOpenAI` wrapper for OpenAI models |
| ☁️ Environment | Google Colab + Google Drive | Notebook runtime & data storage |

---

## 📁 Project Structure

```
📦 rag-pipeline
 ┣ 📓 RAG_PIPELINE.ipynb      # Main notebook with the full pipeline
 ┣ 📂 data
 ┃ ┣ 📄 Python.txt            # Sample text file
 ┃ ┗ 📂 pdfs                  # Folder containing your PDF files
 ┣ 📂 data/vector_store       # Auto-created ChromaDB persistence folder
 ┗ 📝 README.md
```

> 💡 In the notebook the data lives in Google Drive at `/content/drive/MyDrive/RAG/data/`. Adjust the paths if you run locally.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2️⃣ Install dependencies

```bash
pip install langchain langchain-core langchain-community langchain-text-splitters \
            langchain-openai pypdf pymupdf sentence-transformers chromadb scikit-learn
```

### 3️⃣ Add your data

- Put your text file(s) in `data/`
- Put your PDF files in `data/pdfs/`

### 4️⃣ Set your OpenAI API key 🔑

```bash
export OPENAI_API_KEY="your-secret-key-here"      # macOS / Linux
setx OPENAI_API_KEY "your-secret-key-here"        # Windows
```

> 💳 **Note:** The OpenAI API is pay-as-you-go. You need credits in your OpenAI account, since usage is billed by the number of input and output tokens processed.

### 5️⃣ Run the notebook ▶️

Open `RAG_PIPELINE.ipynb` in **Google Colab** or **Jupyter** and run the cells in order.

---

## 🚀 Step-by-Step Walkthrough

### 🔹 Step 1: Setup & Installation
Installs LangChain, PDF processing libraries, the embedding model library, and ChromaDB.

### 🔹 Step 2: Data Ingestion 📄
- **Text file:** `TextLoader` loads `Python.txt` into a LangChain `Document`.
- **PDFs:** `load_all_pdfs()` scans a folder, loads every `.pdf` with `PyPDFLoader`, and merges all pages into one list. It prints the total number of PDFs and pages.

```python
all_pdf_documents = load_all_pdfs()
```

### 🔹 Step 3: Document Chunking ✂️
Large documents are split into smaller overlapping pieces so that each fits comfortably in the LLM context and keeps continuity across boundaries.

```python
def split_docs(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

chunks = split_docs(all_pdf_documents)
```

| Parameter | Default | Meaning |
|---|---|---|
| `chunk_size` | `500` | Max characters per chunk |
| `chunk_overlap` | `50` | Characters shared between neighbouring chunks |

### 🔹 Step 4: Embedding Generation 🧬
The `EmbeddingManager` class wraps a `SentenceTransformer` model (`all-MiniLM-L6-v2`) and converts text into dense **384-dimensional** vectors that capture semantic meaning.

```python
embedding_manager = EmbeddingManager()
```

### 🔹 Step 5: Vector Store (ChromaDB) 💾
The `VectorStoreManager` class:
- Creates a **persistent** ChromaDB client at `data/vector_store`
- Creates or reuses a collection named `pdf_documents`
- Stores each chunk with a unique ID (`doc_<uuid>`), its embedding, text, and metadata (source, page, `doc_index`, `content_length`)

```python
vector_store = VectorStoreManager()

texts = [doc.page_content for doc in chunks]
embeddings = embedding_manager.generate_embeddings(texts)
vector_store.add_documents(chunks, embeddings)
```

### 🔹 Step 6: Retrieval 🔎
The `RAGRetriever` class embeds the user query, runs a similarity search in ChromaDB, converts distance to a similarity score (`1 - distance`), filters by `score_threshold`, and returns ranked results.

```python
rag_retriever = RAGRetriever(embedding_manager, vector_store)
rag_retriever.retrieve("What is encoder decoder")
```

Each result contains: `id`, `document`, `metadata`, `distance`, `similarity_score`, `rank`.

### 🔹 Step 7: LLM Integration & Response Generation 🤖
The retrieved chunks are joined into a context string and combined with the question in a prompt, which is sent to the LLM via LangChain's `ChatOpenAI`.

```python
llm = ChatOpenAI(
    openai_api_key=API_KEY_OPENAI,
    model="gpt-5.4",
    temperature=0.1,
    max_tokens=1024
)

def generate_output(query, retriever, llm, top_k=3):
    results = retriever.retrieve(query, top_k)
    context = "\n".join([doc["document"] for doc in results]) if results else ""

    if not context:
        print("we found no relevant context for the given query")

    prompt = f"""use given context to generate the answer for the query
                Context: {context}
                Query: {query}"""

    response = llm.invoke(prompt)
    return response.content
```

---

## 💻 Usage Example

```python
answer = generate_output("what is RAG?", rag_retriever, llm)
print(answer)
```

**What happens behind the scenes:**
1. ❓ Your question is embedded into a vector
2. 🔎 The top 3 most similar chunks are fetched from ChromaDB
3. 🧩 Those chunks + your question become one prompt
4. 🤖 The LLM answers using that grounded context

---

## 🔧 Configuration Options

| Setting | Where | Default | Description |
|---|---|---|---|
| `chunk_size` | `split_docs()` | `500` | Characters per chunk |
| `chunk_overlap` | `split_docs()` | `50` | Overlap between chunks |
| `model_name` | `EmbeddingManager` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `persist_directory` | `VectorStoreManager` | `data/vector_store` | Where ChromaDB saves data |
| `collection_name` | `VectorStoreManager` | `pdf_documents` | ChromaDB collection name |
| `top_k` | `retrieve()` / `generate_output()` | `5` / `3` | Number of chunks retrieved |
| `score_threshold` | `retrieve()` | `0.0` | Minimum similarity score |
| `temperature` | `ChatOpenAI` | `0.1` | Lower = more factual |
| `max_tokens` | `ChatOpenAI` | `1024` | Max response length |
| `model` | `ChatOpenAI` | `gpt-5.4` | Set this to any OpenAI chat model available on your account |

---

## 🔐 Security Notes

- 🚫 **Never hard-code or commit your API key** to GitHub. The notebook stores it in a variable (`API_KEY_OPENAI`), so make sure it is blank or a placeholder before pushing.
- ✅ Use environment variables, Colab Secrets, or a `.env` file instead:

```python
import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-5.4",
    temperature=0.1,
    max_tokens=1024
)
```

- 🙈 Add this to your `.gitignore`:

```
.env
data/vector_store/
```

---

## 🐞 Known Issues & Fixes

### 1. `add_documents()` indentation

In the notebook, the `self.collection.add(...)` call sits **inside** the `for` loop. That re-adds a growing list on every iteration, which is slow and triggers duplicate-ID warnings. Move it **outside** the loop:

```python
def add_documents(self, documents, embeddings):
    if len(documents) != len(embeddings):
        raise ValueError("num of documents does not match num of embeddings")

    ids, all_metadata, documents_content, embeddings_list = [], [], [], []

    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        ids.append(f"doc_{uuid.uuid4()}")

        metadata = dict(doc.metadata)
        metadata["doc_index"] = i
        metadata["content_length"] = len(doc.page_content)
        all_metadata.append(metadata)

        documents_content.append(doc.page_content)
        embeddings_list.append(embedding.tolist())

    # ✅ add once, after the loop
    self.collection.add(
        ids=ids,
        metadatas=all_metadata,
        documents=documents_content,
        embeddings=embeddings_list
    )

    print("total documents added in vector store =", len(documents_content))
    print("docs in collection:", self.collection.count())
```

### 2. Re-running ingestion duplicates data
Because IDs are random UUIDs, running the ingestion cell twice stores the same chunks twice. Delete `data/vector_store/` (or the collection) before re-ingesting.

### 3. Unused import
`cosine_similarity` from scikit-learn is imported but not used, because ChromaDB already returns distances. It can be removed.

---

## 🔮 Future Improvements

- 🌐 Build a **Streamlit / Gradio** chat UI
- 📚 Support more file types (DOCX, CSV, web pages)
- 🔁 Add **re-ranking** for better retrieval quality
- 💬 Add **conversation memory** for multi-turn chat
- 📎 Show **source citations** (file name and page number) with each answer
- 🧪 Add evaluation metrics (faithfulness, relevance)
- 🏠 Support **local LLMs** (Ollama, Llama) for a fully offline setup
- 🧹 Prevent duplicate ingestion with deterministic IDs

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 🎉

1. 🍴 Fork the repo
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m "Add amazing feature"`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔃 Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License**. Feel free to use, modify, and share. 🆓

---

## 🙌 Acknowledgements

- 🦜 [LangChain](https://www.langchain.com/)
- 💾 [ChromaDB](https://www.trychroma.com/)
- 🧬 [Sentence-Transformers](https://www.sbert.net/)
- 🤖 [OpenAI](https://openai.com/)

---

⭐ **If you found this project helpful, please give it a star!** ⭐
