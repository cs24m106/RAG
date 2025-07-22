# Superset-Ollama SQL Query Pipeline

This project implements a Flask-based application that integrates Apache Superset and Ollama to generate and execute SQL queries on a PostgreSQL database for wireless LTE/4G KPI analysis. It includes two versions of the query generation pipeline (`ollama_sql_ver1.py` and `ollama_sql_ver2.py`) and an analysis script (`analysis.py`) for KPI-related operations.

## Overview

The pipeline converts user queries into SQL queries using Ollama's language model, executes them via Superset's SQL Lab API, and retrieves results from a PostgreSQL database (`public.hrly_kpi_1`). The `analysis.py` script provides additional functionality for KPI correlation, formula extraction, and performance metric retrieval.

### Key Components
- **ollama_sql_ver1.py**: A Flask app that generates SQL queries using the Mistral model via Ollama, executes them in Superset, and returns results. It uses a predefined `TABLE_SCHEMA` and metric interpretation logic to handle KPI queries.
- **ollama_sql_ver2.py**: An enhanced Flask app with an `Attribute` class for schema definition, improved response parsing, and support for custom Ollama models via command-line arguments.
- **analysis.py**: Provides utility functions to analyze KPIs, including correlation extraction, formula parsing, and direct PostgreSQL queries for performance metrics.

## Dependencies

- **Python libraries**:
  - `flask`, `requests`, `json`, `logging`, `pandas`, `sqlalchemy`, `pyyaml`, `openpyxl` (for all scripts)
- **External services**:
  - Ollama (local endpoint at `http://localhost:11434`)
  - Apache Superset (at `http://10.100.80.26:8088`)
  - PostgreSQL (at `10.100.80.23:5532`, database: `analytics`)
- **Files**:
  - `index_ver1.html` (for `ollama_sql_ver1.py`)
  - `index_ver2.html` (for `ollama_sql_ver2.py`)
  - `AllMetrics.xlsx`, `KPI_formula.xlsx`, `hourly_pm_ingestion_config.yml` (for `analysis.py`)

## Environment Setup

1. **Install Python**: Ensure Python 3.8+ is installed.
2. **Install dependencies**:
   ```bash
   pip install flask requests pandas sqlalchemy pyyaml openpyxl
   ```
3. **Set up Ollama**:
   - Install Ollama and start the server locally (`http://localhost:11434`).
   - Pull the `mistral` model (or other models for `ollama_sql_ver2.py`).
4. **Set up Superset**:
   - Ensure Superset is running at `http://10.100.80.23:8088`.
   - Set environment variables:
     ```bash
     export SUPERSET_USERNAME=admin
     export SUPERSET_PASSWORD=admin
     ```
5. **Set up PostgreSQL** (for `analysis.py`):
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
   - Place `AllMetrics.xlsx`, `KPI_formula.xlsx`, and `hourly_pm_ingestion_config.yml` in `/home/techie/logeshv/RAG/work/chromadb/documents/`.
   - Place `index_ver1.html` and `index_ver2.html` in the Flask templates folder.

## Implementation Steps

1. **Prepare Files**:
   - Ensure `AllMetrics.xlsx` (KPI correlations), `KPI_formula.xlsx` (KPI formulas), and `hourly_pm_ingestion_config.yml` (PM-to-table mapping) are in the specified directory.
   - Create `index_ver1.html` and `index_ver2.html` for the Flask app's web interface.

2. **Run the Flask App**:
   - For `ollama_sql_ver1.py`:
     ```bash
     python ollama_sql_ver1.py
     ```
     - Runs on port `5001` with the `mistral` model.
   - For `ollama_sql_ver2.py`:
     ```bash
     python ollama_sql_ver2.py --port 5000 --model mistral
     ```
     - Specify a custom port or Ollama model (e.g., `llama3.2`) if available.
   - Access the web interface at `http://localhost:<port>`.

3. **Script Execution**:
   - **ollama_sql_ver1.py**:
     - Defines the `public.hrly_kpi_1` schema with `TABLE_SCHEMA` and metric interpretation rules.
     - Converts user queries (e.g., "Show top sites by RRC drop rate") into SQL using Ollama's Mistral model.
     - Executes queries via Superset's SQL Lab API and returns results.
   - **ollama_sql_ver2.py**:
     - Uses an `Attribute` class for schema definition and improved response parsing (handles JSON or code block outputs).
     - Supports custom models via `--model` argument and logs detailed query information.
     - Executes queries similarly via Superset.
   - **analysis.py**:
     - `get_top_n_correlated_kpis`: Retrieves top correlated KPIs from `AllMetrics.xlsx`.
     - `get_formula_for_kpi`: Extracts formula JSON for a KPI from `KPI_formula.xlsx`.
     - `extract_pms_from_formula`: Recursively extracts performance metrics (PMs) from formula JSON.
     - `load_pm_table_mapping`: Loads PM-to-table mappings from `hourly_pm_ingestion_config.yml`.
     - `get_pm_value`: Queries PostgreSQL for a PM value by `cell_id` and `timestamp`.
     - `make_timezone_unaware`: Converts datetime columns to timezone-unaware for consistency.

4. **Querying**:
   - Access the web interface and submit queries (e.g., "List worst performing sites by erabdr").
   - The app generates SQL based on the schema and metric interpretation (positive/negative metrics).
   - Results are returned as JSON with the SQL query, results, and (in `ver2`) an explanation.

5. **Customization**:
   - Modify `TABLE_SCHEMA` (`ver1`) or `datebase` (`ver2`) to update the schema.
   - Adjust `positive_metrics` and `negative_metrics` in `ver1` for custom KPI handling.
   - Update `SUPERSET_URL`, `OLLAMA_URL`, or `DATABASE_ID` in the scripts for different environments.
   - For `analysis.py`, modify file paths or database credentials via environment variables.

6. **Output**:
   - Logs detail authentication, query generation, and execution results.
   - The Flask app returns JSON with the SQL query, results, and (in `ver2`) an explanation.
   - `analysis.py` functions return structured data (e.g., lists, JSON, or DataFrames) for KPI analysis.

## Notes
- Ensure Ollama, Superset, and PostgreSQL are running before starting the app.
- Verify environment variables for Superset and PostgreSQL credentials.
- The `create_superset_chart` function is incomplete and requires customization for specific metrics and visualizations.
- For large datasets, adjust the `queryLimit` in `execute_superset_query` (default: 100 in `ver1/ver2`, 10M in `analysis.py`).
- `ollama_sql_ver2.py` is more robust for parsing Ollama responses and supports custom models.