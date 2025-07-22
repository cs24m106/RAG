# Ollama PDF Processing and RAG Pipeline

This project implements a Retrieval-Augmented Generation (RAG) pipeline using Ollama for embeddings and language modeling, integrated with ChromaDB for vector storage and retrieval. It processes documents (PDFs, CSVs, Excel, DOCX) to create a searchable database, enabling semantic search and response generation for queries.

## Overview

The `ollama_pdf.py` script processes documents, generates embeddings using Ollama, and stores them in ChromaDB collections for retrieval. The `preprocess.py` script handles document reading and cleaning, extracting text and metadata from various file formats. Together, they support a RAG pipeline for querying documents with optional query enhancement and term/abbreviation integration.

### Key Components
- **ollama_pdf.py**: Manages the RAG pipeline, including database initialization, document chunking, embedding, retrieval, and response generation.
- **preprocess.py**: Processes DOCX, PDF, CSV, and Excel files, extracting text and metadata while cleaning unwanted sections (e.g., "Contents", tables).

## Dependencies

- **Python libraries**:
  - `langchain_ollama`, `numpy`, `chromadb`, `logging`, `re` (for `ollama_pdf.py`)
  - `python-docx`, `pypdf`, `pandas`, `openpyxl` (for `preprocess.py`)
- **Custom scripts**:
  - `preheader.py`: Custom logging setup
  - `preprocess.py`: Document processing and term/abbreviation extraction
- **External services**:
  - Ollama (local endpoint at `http://localhost:11434`)
  - ChromaDB (running on `127.0.0.1:8000`)

## Environment Setup

1. **Install Python**: Ensure Python 3.8+ is installed.
2. **Install dependencies**:
   ```bash
   pip install langchain-ollama numpy chromadb python-docx pypdf pandas openpyxl
   ```
3. **Set up Ollama**:
   - Install Ollama and start the server locally (`http://localhost:11434`).
   - Pull required models (e.g., `all-minilm` for embeddings, `llama3.2` for chat).
4. **Set up ChromaDB**:
   - Install ChromaDB: `pip install chromadb`
   - Run ChromaDB server:
     ```bash
     chroma run --host 127.0.0.1 --port 8000 --path ./chroma_db &
     ```
5. **Directory structure**:
   - Create a `documents` folder for input files (PDFs, CSVs, Excel, DOCX).
   - Place `preheader.py` and `preprocess.py` in the parent directory of `ollama_pdf.py`.

## Implementation Steps

1. **Prepare Documents**:
   - Place documents (PDFs, CSVs, Excel, DOCX) in the `documents` folder.
   - Ensure filenames indicate content type (e.g., include "formula" for formula-related files).

2. **Run the Script**:
   - Save both `ollama_pdf.py` and `preprocess.py` in the same directory.
   - Execute the main script:
     ```bash
     python ollama_pdf.py
     ```

3. **Script Execution**:
   - **Document Processing** (`preprocess.py`):
     - Reads DOCX, PDF, CSV, and Excel files, extracting text and metadata.
     - Cleans DOCX files by removing specified sections (e.g., "Contents") and optionally tables.
     - Removes "Contents" sections from PDFs using regex-based filtering.
     - For Excel/CSV, extracts row data as enriched strings, dropping empty or numeric-only columns, and captures metadata (e.g., column count, dtypes).
   - **RAG Pipeline** (`ollama_pdf.py`):
     - Initializes ChromaDB with three collections: `KPI_and_PM_lte` (documents), `KPI_and_PM_formulas` (formulas), `KPI_and_PM_analytics` (Excel/CSV analytics).
     - Retrieval Pipeline: 1.forumula --> 2.correlation --> 3.documentations
     - Processes documents into chunks using `RecursiveCharacterTextSplitter` (configurable chunk size/overlap).
     - Generates embeddings with Ollama's `all-minilm` model and stores them in ChromaDB.
     - Runs example queries (e.g., "Describe active E-RABs") and logs retrievals and responses.
     - Press `<ENTER>` to proceed through each query during execution.

4. **Customization**:
   - Modify `initialize_database()` parameters in `ollama_pdf.py` to change:
     - `embed_model`: Embedding model (default: `all-minilm`).
     - `chat_model`: Chat model (default: `llama3.2`).
     - `chunk_size`: Size of text chunks (default: 500).
     - `chunk_overlap`: Overlap between chunks (default: 25).
     - `k_nearest`: Number of retrieved results per collection (default: `(1, 1, 3)` for formulas, Excel/CSV, documents).
     - `terms_definitions` and `abbreviations_definitions`: Add custom terms/abbreviations.
     - `refresh`: Set to `True` to reset the ChromaDB database.
   - Update `questions` list in the `main()` block of `ollama_pdf.py` to test different queries.
   - Adjust `delete_doc_sections()` in `preprocess.py` to modify sections to remove from DOCX files (e.g., add "References" to `titles_to_delete`).

5. **Querying**:
   - The pipeline supports RAG-based querying with optional enhancement (rephrasing queries or adding terms/abbreviations).
   - Use `rag_pipeline(question, retrieve_only=False, should_enhance=True)` for enhanced queries or `retrieve_only=True` for raw retrieval.
   - The `find_terms_and_abbreviations_in_sentence` function in `preprocess.py` enhances queries by matching terms/abbreviations case-sensitively (abbreviations) or case-insensitively (terms).

6. **Output**:
   - Logs detail document processing, chunking, retrievals, and responses.
   - For each query, retrieves relevant document chunks and, if not in `retrieve_only` mode, generates a response using the LLM.
   - Metadata (e.g., file properties, column counts for Excel/CSV) is stored and logged.

## Notes
- Ensure Ollama and ChromaDB servers are running before executing the script.
- Adjust `OLLAMA_URL` or ChromaDB host/port in `ollama_pdf.py` if using a different configuration.
- For large document sets, increase `chunk_size` or adjust `k_nearest` for better performance.
- The `preprocess.py` script handles errors gracefully, logging issues and skipping corrupted files (file deletion is commented out for safety).