# RAG and SQL Query Pipeline

This project integrates a Retrieval-Augmented Generation (RAG) pipeline with SQL query generation and execution for analyzing wireless LTE/4G KPIs. The `pipeline.py` script combines document retrieval (via ChromaDB and Ollama) with SQL query generation (via PostgreSQL and Superset) to answer complex queries using both document context and database data.

## Overview

The `pipeline.py` script serves as the main module, orchestrating:
- **Document Retrieval**: Uses `ollama_pdf.py` to retrieve relevant document chunks from ChromaDB.
- **SQL Query Generation**: Generates SQL queries for the `public.hrly_kpi_1` table using Ollama, enhanced with document context.
- **Query Execution**: Executes SQL queries directly on a PostgreSQL database using `analysis.py`.
- **Logging**: Implements custom logging via `preheader.py` for console and file output.

## Dependencies

- **Python libraries**:
  - `pandas`, `json`, `logging`
  - From `ollama_pdf.py`: `langchain_ollama`, `numpy`, `chromadb`, `re`
  - From `analysis.py`: `requests`, `sqlalchemy`, `pyyaml`, `openpyxl`
  - From `preprocess.py`: `python-docx`, `pypdf`, `pandas`, `openpyxl`
- **Custom scripts**:
  - `preheader.py`: Custom logging setup
  - `preprocess.py`: Document processing and term/abbreviation extraction
  - `ollama_pdf.py`: RAG pipeline for document retrieval
  - `ollama_sql_ver2.py`: SQL query generation (provides `datebase` schema)
  - `analysis.py`: PostgreSQL query execution
- **External services**:
  - Ollama (local endpoint: `http://localhost:11434`)
  - PostgreSQL (host: `10.100.80.23`, port: `5532`, database: `analytics`)
  - ChromaDB (host: `127.0.0.1`, port: `8000`)

## Environment Setup

1. **Install Python**: Ensure Python 3.8+ is installed.
2. **Install dependencies**:
   ```bash
   pip install pandas langchain-ollama numpy chromadb python-docx pypdf openpyxl requests sqlalchemy pyyaml
   ```
3. **Set up Ollama**:
   - Install Ollama and start the server (`http://localhost:11434`).
   - Pull models: `mistral`, `all-minilm`, `llama3.2`.
4. **Set up ChromaDB**:
   - Install ChromaDB: `pip install chromadb`
   - Run ChromaDB server:
     ```bash
     chroma run --host 127.0.0.1 --port 8000 --path ./chroma_db &
     ```
     - host: ip address on which chromaDB run on
     - port: default 8000 (optional)
     - path if using persistant database (optional)
     - `&` means to run in backgroud (optional)
5. **Set up PostgreSQL**:
   - Ensure PostgreSQL is running at `10.100.80.23:5532` with database `analytics`.
   - Set environment variables:
     ```bash
     export PG_USERNAME=db-admin
     export PG_PASSWORD=db-admin
     export PG_HOST=10.100.80.23
     export PG_PORT=5532
     export PG_DATABASE=analytics
     ```
6. **Directory structure**:
   - Place `preheader.py` in the same directory as `pipeline.py`.
   - Place `preprocess.py` and `ollama_pdf.py` in the `chromaDatabase` sub-directory. Within this sub-dir:
      - Create a `documents` folder for input files (PDFs, CSVs, Excel, DOCX).
      - Create a `chroma_db` folder for the ChromaDB database (needed if using persistant path)
   - Place `ollama_sql_ver2.py` and `analysis.py` in the `postgreSuperset` sub-directory.

## Implementation Steps

1. **Prepare Documents**:
   - Place documents (PDFs, CSVs, Excel, DOCX) in the `documents` folder.
   - Ensure filenames indicate content type (e.g., include "formula" for formula-related files).

2. **Run the Script**:
   - Save `pipeline.py`, `preheader.py`, `preprocess.py`, `ollama_pdf.py`, `ollama_sql_ver2.py`, and `analysis.py` in the same directory.
   - Execute the main script:
     ```bash
     python pipeline.py
     ```

3. **Script Execution**:
   - **Initialization**:
     - Calls `initialize_database()` and `prepare_dataset()` from `ollama_pdf.py` to set up ChromaDB collections (`KPI_and_PM_lte`, `KPI_and_PM_formulas`, `KPI_and_PM_analytics`) and process documents.
   - **Query Processing**:
     - Iterates through predefined `questions` (e.g., "What are the worst 10 performing sites based on RRCDR?").
     - Retrieves document context using `rag_pipeline(retrieve_only=True)` from `ollama_pdf.py`.
     - Generates SQL queries using `fetch_sql_generation()`, combining the `public.hrly_kpi_1` schema (`datebase` from `ollama_sql_ver2.py`) and retrieved context.
     - Executes SQL queries via `run_sql_query()`, which uses `execute_pg_query()` from `analysis.py` to query PostgreSQL.
   - **Output**:
     - Logs document processing, retrieval, SQL generation, and query results.
     - Displays results as pandas DataFrames (if data is returned).
     - Pauses after each query (press `<ENTER>` to continue).

4. **Customization**:
   - Modify `initialize_database()` parameters in `ollama_pdf.py` (called by `pipeline.py`):
     - `embed_model`: Embedding model (default: `all-minilm`).
     - `chat_model`: Chat model (default: `llama3.2`).
     - `chunk_size`: Text chunk size (default: 500).
     - `chunk_overlap`: Chunk overlap (default: 25).
     - `k_nearest`: Retrieved results per collection (default: `(1, 1, 3)`).
     - `terms_definitions`, `abbreviations_definitions`: Add custom terms/abbreviations.
     - `refresh`: Set to `True` to reset ChromaDB.
   - Update the `questions` list in `pipeline.py` to test new queries.
   - Modify `table_name` or `table_dict` in `fetch_sql_generation()` to query different tables.

5. **Querying**:
   - Queries are enhanced with document context from ChromaDB and executed against PostgreSQL.
   - The `fetch_sql_generation()` function ensures SQL queries use only valid `public.hrly_kpi_1` columns and include top 3 correlated attributes (if applicable).
   - Results are returned as DataFrames, with error handling for invalid queries or empty results.

6. **Output**:
   - Logs are written to the console and a file (e.g., `pipeline.log`) using `preheader.py`'s custom logging.
   - Each query logs the question, retrieved context, generated SQL, and results.
   - Similarity scores are logged if expected answers are provided (currently empty `answers` list).

## Notes
- Ensure Ollama, ChromaDB, and PostgreSQL are running before execution.
- Adjust `OLLAMA_URL`, ChromaDB host/port, or PostgreSQL credentials if using different configurations.
- The `preheader.py` script creates a log file in the same directory, overwriting previous logs.
- For large datasets, adjust `chunk_size`, `k_nearest`, or PostgreSQL query limits in `analysis.py`.
- The pipeline assumes `public.hrly_kpi_1` exists in the PostgreSQL database with the schema defined in `ollama_sql_ver2.py`.
- Please refer to log files of some the codes, that are added to show how it works w.r.t inputs given as well