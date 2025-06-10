from flask import Flask, request, jsonify, render_template
import requests
import json
import os
from datetime import datetime, timedelta
import logging

app = Flask(__name__)

# Configuration
# SUPERSET_URL = "http://localhost:8088"  # Superset instance URL
SUPERSET_URL = "http://10.100.80.26:8088"
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

# Helper function to query Ollama (Mistral) for SQL generation
def query_ollama(prompt):
    payload = {
        "model": "mistral",
        "prompt": f"""
        You are a data analyst using Apache Superset with a PostgreSQL database.
        You are also an expert in Wireless LTE/4G technology.
        The database has a table named 'public.hrly_kpi_1' with columns: 
        enb_id (string), 
        sector_id (string), 
        totalpayloadgb (numeric),
        rrcsrnom(numeric),
        rrc_connection_success_rate_all (numeric),
        and many others.
        Given the user query, generate a valid SQL query.
        User query: "{prompt}"
        Provide only the SQL query, no explanations.'
        The output of the query needs to be presented as a table.
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
    return render_template("index.html")

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
    if not sql_query:
        return jsonify({"error": "Failed to generate SQL query"}), 500
    
    # Execute SQL query in Superset
    results = execute_superset_query(sql_query)
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
    app.run(debug=True, port=5000)