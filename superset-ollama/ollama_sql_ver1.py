from flask import Flask, request, jsonify, render_template
import requests
import json
import os
from datetime import datetime, timedelta
import logging

TABLE_SCHEMA = [
    {
    "name": "time",
    "type": "timestamp with time zone",
    "description": "Timestamp indicating when the KPI measurements were recorded."
    },
    {
        "name": "enb_id",
        "type": "integer",
        "description": "Unique identifier for the eNodeB (cell site)."
    },
    {
        "name": "sector_id",
        "type": "integer",
        "description": "Identifier for the sector within the cell site."
    },
    {
        "name": "rrc_connection_success_rate_all",
        "type": "numeric",
        "description": "RRC connection success rate across all attempts."
    },
    {
        "name": "intra_frequency_handover_out_success_rate",
        "type": "numeric",
        "description": "Success rate for intra-frequency handovers from this sector."
    },
    {
        "name": "inter_frequency_handover_out_success_rate",
        "type": "numeric",
        "description": "Success rate for inter-frequency handovers from this sector."
    },
    {
        "name": "totalpayloadgb",
        "type": "numeric",
        "description": "Total data payload in gigabytes."
    },
    {
        "name": "rrcsrdenom",
        "type": "integer",
        "description": "Denominator for RRC setup success rate."
    },
    {
        "name": "rrcsrnom",
        "type": "integer",
        "description": "Numerator for RRC setup success rate."
    },
    {
        "name": "rrcdr",
        "type": "numeric",
        "description": "RRC drop rate. Higher values indicate worse performance."
    },
    {
        "name": "erabsrnom",
        "type": "integer",
        "description": "Numerator for ERAB setup success rate."
    },
    {
        "name": "erabsrdenom",
        "type": "integer",
        "description": "Denominator for ERAB setup success rate."
    },
    {
        "name": "erabsr",
        "type": "numeric",
        "description": "ERAB setup success rate."
    },
    {
        "name": "pserabdrnom",
        "type": "integer",
        "description": "Numerator for PS ERAB drop rate."
    },
    {
        "name": "pserabdrdenom",
        "type": "integer",
        "description": "Denominator for PS ERAB drop rate."
    },
    {
        "name": "pshonom",
        "type": "integer",
        "description": "Numerator for PS HO success rate."
    },
    {
        "name": "pshodenom",
        "type": "integer",
        "description": "Denominator for PS HO success rate."
    },
    {
        "name": "pshosr",
        "type": "numeric",
        "description": "PS HO success rate."
    },
    {
        "name": "avgperusercellthpdlmbps",
        "type": "numeric",
        "description": "Average downlink throughput per user in Mbps."
    },
    {
        "name": "avgperusercellthpulmbps",
        "type": "numeric",
        "description": "Average uplink throughput per user in Mbps."
    },
    {
        "name": "erabdrnom",
        "type": "integer",
        "description": "Numerator for ERAB drop rate."
    },
    {
        "name": "erabdrdenom",
        "type": "integer",
        "description": "Denominator for ERAB drop rate."
    },
    {
        "name": "erabdr",
        "type": "numeric",
        "description": "ERAB drop rate."
    },
    {
        "name": "iratghosr",
        "type": "numeric",
        "description": "IRAT GHO success rate."
    },
    {
        "name": "iratuhosr",
        "type": "numeric",
        "description": "IRAT UHO success rate."
    },
    {
        "name": "csfbpct",
        "type": "numeric",
        "description": "CS fallback percentage."
    },
    {
        "name": "voltetrfcerl",
        "type": "numeric",
        "description": "VoLTE RF call establishment rate."
    },
    {
        "name": "rrccongnom",
        "type": "integer",
        "description": "Numerator for RRC congestion rate."
    },
    {
        "name": "voerabdropnom",
        "type": "integer",
        "description": "Numerator for VoERAB drop rate."
    },
    {
        "name": "voerabdropdenom",
        "type": "integer",
        "description": "Denominator for VoERAB drop rate."
    },
    {
        "name": "voerabdrop",
        "type": "numeric",
        "description": "VoERAB drop rate."
    },
    {
        "name": "voerabsrnom",
        "type": "integer",
        "description": "Numerator for VoERAB setup rate."
    },
    {
        "name": "voerabsrdenom",
        "type": "integer",
        "description": "Denominator for VoERAB setup rate."
    },
    {
        "name": "voerabsr",
        "type": "numeric",
        "description": "VoERAB setup success rate."
    },
    {
        "name": "s1hosr",
        "type": "numeric",
        "description": "S1 handover success rate."
    },
    {
        "name": "x2hosr",
        "type": "numeric",
        "description": "X2 handover success rate."
    },
    {
        "name": "cellavailability",
        "type": "numeric",
        "description": "Percentage indicating availability of the cell."
    },
    {
        "name": "dlpktlossratenom",
        "type": "integer",
        "description": "Numerator for downlink packet loss rate."
    },
    {
        "name": "dlpktlossratedenom",
        "type": "integer",
        "description": "Denominator for downlink packet loss rate."
    },
    {
        "name": "ulpktlossratenom",
        "type": "integer",
        "description": "Numerator for uplink packet loss rate."
    },
    {
        "name": "ulpktlossratedenom",
        "type": "integer",
        "description": "Denominator for uplink packet loss rate."
    },
    {
        "name": "data_volume_dl_gb",
        "type": "numeric",
        "description": "Downlink data volume in gigabytes."
    },
    {
        "name": "data_volume_ul_gb",
        "type": "numeric",
        "description": "Uplink data volume in gigabytes."
    },
    {
        "name": "allhoatt",
        "type": "integer",
        "description": "Total number of handover attempts."
    },
    {
        "name": "erabreleutrangenreasonqci1",
        "type": "integer",
        "description": "Count of ERAB releases due to range reason for QCI 1."
    }
]


metric_interpretation = """
    ### Metric Interpretation

    The following are considered **positive  metrics**, meaning higher values indicate better performance:
    - rrc_connection_success_rate_all
    - intra_frequency_handover_out_success_rate
    - inter_frequency_handover_out_success_rate
    - totalpayloadgb
    - data_volume_dl_gb
    - data_volume_ul_gb
    - allhoatt

    The following are considered **negative metrics**, meaning lower values indicate better performance:
    - rrcdr
    - erabdr
    - dlpktlossratenom
    - dlpktlossratedenom
    - ulpktlossratenom
    - ulpktlossratedenom

    When identifying worst performers:
    - For positive metrics: low values are worse
    - For negative metrics: high values are worse

    When identifying best performers:
    - For positive metrics: high values are better
    - For negative metrics: low values are better
    """
positive_metrics = [
    "rrc_connection_success_rate_all",
    "intra_frequency_handover_out_success_rate",
    "inter_frequency_handover_out_success_rate",
    "totalpayloadgb",
    "data_volume_dl_gb",
    "data_volume_ul_gb",
    "allhoatt"
]

negative_metrics = [
    "rrcdr",
    "erabdr",
    "dlpktlossratenom",
    "dlpktlossratedenom",
    "ulpktlossratenom",
    "ulpktlossratedenom",
    "voerabdrop",
    "pserabdr"
]

QUERY_GENERATION_INSTRUCTIONS = f"""
    - When query is about a **site**, use GROUP BY enb_id
    - Always qualify column names with table name to avoid ambiguity
    - Use a CTE or subquery if required. In case of using CTE, carefully handle column names to avoid conflicts
    - Carefully handle the aliases to avoid conflicts, especially when using aggregates or subqueries
    - Use aggregates where needed (e.g., AVG for metrics like rrcdr per site)
    - ALWAYS use only actual column names. NEVER ASSUME columns that don't exist.

    ### Metric Interpretation Logic (Step-by-step)

    The following are considered **positive metrics**, meaning higher values indicate better performance:
    - {', '.join(positive_metrics)}

    The following are considered **negative metrics**, meaning lower values indicate better performance:
    - {', '.join(negative_metrics)}

    1. Identify whether the KPI in the user query is a **positive** or **negative** metric:
        - Positive Metrics: Higher values indicate better performance
            Example: rrc_connection_success_rate_all, intra_frequency_handover_out_success_rate
        - Negative Metrics: Lower values indicate better performance
            Example: rrcdr, erabdr, voerabdrop

    2. Based on the type of metric, follow the appropriate rules for generating queries:

        A. If it is a **positive metrics**:
            - Worst performers are those with the **lowest values**
                → Use ASCENDING sort to get them
            - Best performers are those with the **highest values**
                → Use DESCENDING sort to get them

        B. If it is a **negative metrics**:
            - Worst performers are those with the **highest values**
                → Use DESCENDING sort to get them
            - Best performers are those with the **lowest values**
                → Use ASCENDING sort to get them

    3. When identifying **worst performers**:
        - Filter out zero or null values when relevant to avoid skewing results
        - Sort by metric:
            - For **positive metrics**: ASC
            - For **negative metrics**: DESC
        - Then sort final output in requested order (ASC or DESC)

    4. When identifying **best performers**:
        - Filter out zero or null values when relevant to avoid skewing results
        - Sort by metric:
            - For **positive metrics**: DESC
            - For **negative metrics**: ASC
        - Then sort final output in requested order (ASC or DESC)

    5. When the prompt is about a site, use enb_id to group by site
    6. Before returning the SQL query, ENSURE it is syntactically correct and valid
"""

app = Flask(__name__)

# Configuration
# SUPERSET_URL = "http://localhost:8088"  # Superset instance URL
SUPERSET_URL = "http://10.100.80.23:8088" #sys.26:port def
OLLAMA_URL = "http://localhost:11434/api/generate"  # Ollama local endpoint
DATABASE_ID = 2  # Superset database ID for SQL Lab
SUPERSET_USERNAME = os.getenv("SUPERSET_USERNAME", "admin")  # Set in environment
SUPERSET_PASSWORD = os.getenv("SUPERSET_PASSWORD", "admin")  # Set in environment

# Token storage
access_token = None
refresh_token = None
token_expiry = None
session = requests.Session()  # Maintain session for cookies

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Helper function to authenticate with Superset and get tokens
def authenticate_superset():
    global access_token, refresh_token, token_expiry
    url = f"{SUPERSET_URL}/api/v1/security/login"
    payload = {
        "username": SUPERSET_USERNAME,
        "password": SUPERSET_PASSWORD,
        "provider": "db",
        "refresh": True
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = session.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            access_token = data.get("access_token")
            refresh_token = data.get("refresh_token")
            token_expiry = datetime.now() + timedelta(hours=1)  # Adjust based on Superset config
            logger.debug("Successfully authenticated with Superset")
            logger.debug(f"Session cookies: {session.cookies.get_dict()}")
            return access_token
        else:
            logger.error(f"Authentication failed: {response.status_code} {response.text}")
            return None
    except requests.RequestException as e:
        logger.error(f"Authentication error: {e}")
        return None

# Helper function to refresh access token
def refresh_superset_token():
    global access_token, refresh_token, token_expiry
    if not refresh_token:
        logger.error("No refresh token available, re-authenticating")
        return authenticate_superset()
    
    url = f"{SUPERSET_URL}/api/v1/security/refresh"
    headers = {
        "Authorization": f"Bearer {refresh_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = session.post(url, headers=headers)
        if response.status_code == 200:
            access_token = response.json().get("access_token")
            token_expiry = datetime.now() + timedelta(hours=1)
            logger.debug("Successfully refreshed access token")
            logger.debug(f"Session cookies: {session.cookies.get_dict()}")
            return access_token
        else:
            logger.error(f"Token refresh failed: {response.status_code} {response.text}")
            return authenticate_superset()
    except requests.RequestException as e:
        logger.error(f"Token refresh error: {e}")
        return authenticate_superset()

# Helper function to get CSRF token
def get_csrf_token():
    global access_token, token_expiry
    if not access_token or datetime.now() >= token_expiry:
        logger.debug("Access token missing or expired, refreshing")
        access_token_new = refresh_superset_token()
        if not access_token_new:
            logger.error("Failed to obtain access token")
            return None
        access_token = access_token_new
    
    url = f"{SUPERSET_URL}/api/v1/security/csrf_token/"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = session.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get("result")
        elif response.status_code == 401:
            logger.debug("Access token invalid, refreshing")
            access_token_new = refresh_superset_token()
            if access_token_new:
                access_token = access_token_new
                headers["Authorization"] = f"Bearer {access_token}"
                response = session.get(url, headers=headers)
                if response.status_code == 200:
                    return response.json().get("result")
            logger.error(f"Failed to get CSRF token after refresh: {response.status_code} {response.text}")
            return None
        else:
            logger.error(f"Failed to get CSRF token: {response.status_code} {response.text}")
            return None
    except requests.RequestException as e:
        logger.error(f"CSRF token fetch error: {e}")
        return None

def query_ollama(prompt):
    # Build column info string from TABLE_SCHEMA
    column_info = "\n".join([
        f"- {col['name']} ({col['type']}): {col['description']}"
        for col in TABLE_SCHEMA
    ])

    payload = {
        "model": "mistral",
        "prompt": f"""
        ### Role
        You are an expert data analyst working with Apache Superset and PostgreSQL.
        You are also an expert in Wireless LTE/4G technology.

        ### Database Schema
        The database has a table named 'public.hrly_kpi_1' with the following columns:
        
        {column_info}

        Instructions for query generation:
        {QUERY_GENERATION_INSTRUCTIONS}

        ### Task
        Generate a valid PostgreSQL SQL query based on the user input below.
        Return ONLY the SQL query — no markdown, no explanation, no extra text.


        ### User Query
        "{prompt}"

        ### Output Format
        - Output ONLY the SQL query. No explanations. No extra text or characters.
        - DO NOT USE MARKDOWN.
        - Must be syntactically correct SQL
        - Use proper aliases and qualify column names when needed
        - Return only the requested columns in a tabular format
        """,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")
        else:
            logger.error(f"Ollama request failed: {response.status_code} {response.text}")
            return None
    except requests.RequestException as e:
        logger.error(f"Ollama error: {e}")
        return None

# Helper function to execute SQL query via Superset SQL Lab API
def execute_superset_query(sql_query):
    global access_token, token_expiry
    csrf_token = get_csrf_token()
    if not csrf_token:
        logger.error("No CSRF token available")
        return None
    
    if not access_token:
        logger.error("No access token available")
        return None
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-CSRF-Token": csrf_token
    }
    payload = {
        "database_id": DATABASE_ID,
        "sql": sql_query,
        "queryLimit": 100
    }
    
    try:
        response = session.post(f"{SUPERSET_URL}/api/v1/sqllab/execute/", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json().get("data", [])
        elif response.status_code == 401:
            logger.debug("Access token invalid, refreshing")
            access_token_new = refresh_superset_token()
            if access_token_new:
                access_token = access_token_new
                headers["Authorization"] = f"Bearer {access_token}"
                # Fetch new CSRF token after refreshing access token
                csrf_token = get_csrf_token()
                if not csrf_token:
                    logger.error("Failed to get new CSRF token after refresh")
                    return None
                headers["X-CSRF-Token"] = csrf_token
                response = session.post(f"{SUPERSET_URL}/api/v1/sqllab/execute/", headers=headers, json=payload)
                if response.status_code == 200:
                    return response.json().get("data", [])
            logger.error(f"Superset query failed after refresh: {response.status_code} {response.text}")
            return None
        else:
            logger.error(f"Superset query failed: {response.status_code} {response.text}")
            return None
    except requests.RequestException as e:
        logger.error(f"Superset query error: {e}")
        return None

# Route for chat interface
@app.route("/")
def index():
    global access_token, token_expiry
    # Initialize access token on app startup
    if not access_token or datetime.now() >= token_expiry:
        if not authenticate_superset():
            logger.error("Failed to initialize access token on startup")
            # Render index.html anyway to allow frontend to load
    return render_template("index_ver1.html")

# Route to handle chat queries
@app.route("/chat", methods=["POST"])
def chat():
    global access_token, token_expiry
    # Ensure access token is initialized
    if not access_token or datetime.now() >= token_expiry:
        if not authenticate_superset():
            return jsonify({"error": "Failed to authenticate with Superset"}), 500
    
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    # Generate SQL using Mistral via Ollama
    sql_query = query_ollama(user_query)
    print(f"Generated SQL query: {sql_query}")
    if not sql_query:
        return jsonify({"error": "Failed to generate SQL query"}), 500
    
    # Execute SQL query in Superset
    results = execute_superset_query(sql_query)

    print(f"Superset results: {results}")

    # create_superset_chart(sql_query=sql_query)
    if not results:
        return jsonify({"error": "Failed to execute query in Superset"}), 500

    return jsonify({
        "sql_query": sql_query,
        "results": results
    })

def create_superset_chart(sql_query, viz_type="bar"):
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    payload = {
        "datasource_id": DATABASE_ID,
        "viz_type": viz_type,
        "params": {"metrics": ["SUM(amount)"], "groupby": ["region"]}
    }
    response = requests.post(f"{SUPERSET_URL}/api/v1/chart/", headers=headers, json=payload)
    return response.json().get("id") if response.status_code == 201 else None

if __name__ == "__main__":
    app.run(debug=True, port=5001)