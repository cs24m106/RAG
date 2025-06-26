from flask import Flask, request, jsonify, render_template
import requests
import json
import os
from datetime import datetime, timedelta
import logging
import argparse

app = Flask(__name__)

# Database Attributes
class Attribute:
    def __init__(self, name: str, type: str = "any", description: str = ""):
        self.name = name
        self.type = type
        self.description = description
    
    def __str__(self):
        return f"\'{self.name}\' ({self.type}): {self.description}"
    
    def get(self):
        return f"{self.name} ({self.type}), -- {self.description}"

datebase = [
    Attribute(name='time', type='TIMESTAMP WITH TIME ZONE', description="Timestamp of the record"),
    Attribute(name='erabreleutrangenreasonqci1', type='NUMERIC', description="ERAB releases due to EUTRAN generated reason for QCI 1"),
    Attribute(name='s1hosr', type='NUMERIC', description="S1 handover success rate"),
    Attribute(name='x2hosr', type='NUMERIC', description="X2 handover success rate"),
    Attribute(name='intra_frequency_handover_out_success_rate', type='NUMERIC', description="Intra-frequency handover out success rate"),
    Attribute(name='rrc_connection_success_rate_all', type='NUMERIC', description="RRC connection success rate (all)"),
    Attribute(name='data_volume_dl_gb', type='NUMERIC', description="Downlink data volume (GB)"),
    Attribute(name='data_volume_ul_gb', type='NUMERIC', description="Uplink data volume (GB)"),
    Attribute(name='cellavailability', type='NUMERIC', description="Cell availability"),
    Attribute(name='totalpayloadgb', type='NUMERIC', description="Total payload (GB)"),
    Attribute(name='allhoatt', type='NUMERIC', description="All handover attempts"),
    Attribute(name='avgperusercellthpdlmbps', type='NUMERIC', description="Average per user cell throughput (DL, Mbps)"),
    Attribute(name='avgperusercellthpulmbps', type='NUMERIC', description="Average per user cell throughput (UL, Mbps)"),
    Attribute(name='voerabdropdenom', type='NUMERIC', description="VoLTE ERAB drop denominator"),
    Attribute(name='pserabdrdenom', type='NUMERIC', description="PS ERAB drop denominator"),
    Attribute(name='erabdrdenom', type='NUMERIC', description="ERAB drop denominator"),
    Attribute(name='voerabdropnom', type='NUMERIC', description="VoLTE ERAB drop numerator"),
    Attribute(name='pserabdrnom', type='NUMERIC', description="PS ERAB drop numerator"),
    Attribute(name='erabdrnom', type='NUMERIC', description="ERAB drop numerator"),
    Attribute(name='voerabdrop', type='NUMERIC', description="VoLTE ERAB drop rate"),
    Attribute(name='erabdr', type='NUMERIC', description="ERAB drop rate"),
    Attribute(name='voerabsrdenom', type='NUMERIC', description="VoLTE ERAB setup success denominator"),
    Attribute(name='erabsrdenom', type='NUMERIC', description="ERAB setup success denominator"),
    Attribute(name='voerabsrnom', type='NUMERIC', description="VoLTE ERAB setup success numerator"),
    Attribute(name='erabsrnom', type='NUMERIC', description="ERAB setup success numerator"),
    Attribute(name='voerabsr', type='NUMERIC', description="VoLTE ERAB setup success rate"),
    Attribute(name='erabsr', type='NUMERIC', description="ERAB setup success rate"),
    Attribute(name='dlpktlossratedenom', type='NUMERIC', description="Downlink packet loss rate denominator"),
    Attribute(name='dlpktlossratenom', type='NUMERIC', description="Downlink packet loss rate numerator"),
    Attribute(name='ulpktlossratedenom', type='NUMERIC', description="Uplink packet loss rate denominator"),
    Attribute(name='ulpktlossratenom', type='NUMERIC', description="Uplink packet loss rate numerator"),
    Attribute(name='iratghosr', type='NUMERIC', description="Inter-RAT GSM handover success rate"),
    Attribute(name='iratuhosr', type='NUMERIC', description="Inter-RAT UMTS handover success rate"),
    Attribute(name='csfbpct', type='NUMERIC', description="CSFB (Circuit Switched Fallback) percentage"),
    Attribute(name='rrccongnom', type='NUMERIC', description="RRC connection congestion numerator"),
    Attribute(name='rrcdr', type='NUMERIC', description="RRC drop rate"),
    Attribute(name='voltetrfcerl', type='NUMERIC', description="VoLTE transfer call establishment release"),
    Attribute(name='rrcsrdenom', type='NUMERIC', description="RRC success rate denominator"),
    Attribute(name='rrcsrnom', type='NUMERIC', description="RRC success rate numerator"),
    Attribute(name='pshodenom', type='NUMERIC', description="PS handover denominator"),
    Attribute(name='pshonom', type='NUMERIC', description="PS handover numerator"),
    Attribute(name='pshosr', type='NUMERIC', description="PS handover success rate"),
    Attribute(name='enb_id', type='INTEGER', description="eNodeB ID defining the actual site"),
    Attribute(name='sector_id', type='INTEGER', description="Sector ID defining a subsection of the eNodeB"),
]

# Definitions
attributes = "\t"+"\n\t".join([str(attr.get()) for attr in datebase]) # use schema form for sqlcoder
definitions = {
    'drop_rate': "higher values indicate more dropped connections, which is considered as bad performance",
    'success_rate': "higher values indicate more successful connections, which is considered as good performance",
    #'val_denom': "if you notice the suffix 'denom' to any attribute, then it is the denominator part of the actual value",
    #'nom': "if you notice the suffix 'nom' to any attribute, then it is the numerator part of the actual value",
    #'actual_val': "the actual value of the respective attribute is calculated by (val_nom/val_denom)"
}

# Model List available from https://ollama.com/library
model = "mistral"  # default model taken for now
MODEL_URL = "http://localhost:11434/api/tags" # Ollama API endpoint to get available models

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

# Helper function to extract SQL from the generated response
def refractor_response(generated):
    logger.debug(f"Need to refractor (into a parsable dict form) the generated response: {generated}")
    output = {}
    output['sql_query'] = generated
    output['explanation'] = None

    # if the generated response follows the given format
    start = generated.find('{') 
    end = generated.find('}', start) + 1 # sqlcoder gives lots of unnecessary text, lets stick with first '{' & '}'
    if start != -1 and end != -1:
        refractored = generated[start:end]    
        try:
            output = json.loads(refractored)
            if output == {}:
                raise ValueError(f"Ollama response is empty or invalid JSON. \n Response Generated: \n{generated}")
        except Exception as e:
            logger.error(f"Failed to parse Ollama response as JSON: {e}\nRefractored Response: \n{refractored}")
    
    if output and 'sql_query' in output: # return if 'sql_query' key is found in parsed dict
        return output
    
    # Try to extract SQL if sqlcoder returns code block formatting (e.g., ```sql ... ```)
    if generated:
        code_start = generated.find("```")
        if code_start != -1:
            # Try to find the next code block end
            code_end = generated.find("```", code_start + 3)
            if code_end != -1:
                # Extract the code block content
                sql_block = generated[code_start + 3:code_end].strip()
                # Remove 'sql' language tag if present
                if sql_block.lower().startswith("sql"):
                    sql_block = sql_block[3:].strip()
                explanation = (generated[:code_start] + generated[code_end + 3:]).strip()
                output = {
                    "sql_query": sql_block,
                    "explanation": explanation
                }
                return output
    
    return output

# Helper function to query Ollama (Mistral) for SQL generation
def query_ollama(prompt):
    payload = {
        "model": model,
        "prompt": f"""
### Instructions:
You are a data analyst using Apache Superset with a PostgreSQL database.
You are also an expert in Wireless LTE/4G technology.
Your task is to convert a question into a SQL query, given a Postgres database schema.
Adhere to these rules:
- **Deliberately go through the question and database schema word by word** to appropriately answer the question
- **Use Table Aliases** to prevent ambiguity (if the query is ambiguous, give the most relavent answer)
- **Use the Table Name** as provided withing the quotes as it is, do not change it
- **Use only the attributes, i.e. column names provived** to form the sql query, add the schmema to your memory to answer carefully
- Try to **avoid using null values for comparison** based queries, unless explicitly asked
- **Follow the given Response Format strictly**, no additional texts or comments or examples outside the format is entertained
- Just return the response in **plain text** with no markdown based formatting

### Input:
This query will run on a database whose schema is represented in this string:
CREATE TABLE 'public.hrly_kpi_1' (
{attributes}
);

Definitions: {definitions}
        
Given the user query, Generate a SQL query that answers the question based on the table schemas and definitions provided to understand the analogy between the attributes.
User query: "{prompt}"

### Response Format:
{{
    "sql_query": "give the sql query as plain text without any formatting, and no comments in between it",
    "explanation": "Reasoning: A brief explanation of the SQL query and how it relates to the user query",
}}
        """,
        "stream": False
    }
    generated = None
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            generated = result.get("response", "")
        else:
            logger.error(f"Ollama request failed: {response.status_code} {response.text}")
    except requests.RequestException as e:
        logger.error(f"Ollama error: {e}")
    
    output = refractor_response(generated)    
    logger.debug(f"Ollama Final Output: {output}")
    return output['sql_query'], output['explanation'], payload

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
    
    data = None
    text = ""
    try:
        response = session.post(f"{SUPERSET_URL}/api/v1/sqllab/execute/", headers=headers, json=payload)
        if response.status_code == 200:
            data =  response.json().get("data", [])
        elif response.status_code == 401:
            logger.debug("Access token invalid, refreshing")
            access_token_new = refresh_superset_token()
            if access_token_new:
                access_token = access_token_new
                headers["Authorization"] = f"Bearer {access_token}"
                # Fetch new CSRF token after refreshing access token
                csrf_token = get_csrf_token()
                if not csrf_token:
                    text += "Failed to get new CSRF token after refresh. "
                    logger.error(text)
                else:
                    headers["X-CSRF-Token"] = csrf_token
                    response = session.post(f"{SUPERSET_URL}/api/v1/sqllab/execute/", headers=headers, json=payload)
                    if response.status_code == 200:
                        data = response.json().get("data", [])
                    else:
                        text += "Superset query failed after refresh. "
                        logger.error(f"{text} {response.status_code} {response.text}")
        else:
            text += "Superset query failed.  "
            logger.error(f"{text} {response.status_code} {response.text}")
    except requests.RequestException as e:
        logger.error(f"Superset query error: {e}")
    
    text += f"{response.status_code} {response.text}"
    return data, text

# Route for chat interface
@app.route("/")
def index():
    global access_token, token_expiry
    # Initialize access token on app startup
    if not access_token or datetime.now() >= token_expiry:
        if not authenticate_superset():
            logger.error("Failed to initialize access token on startup!")
            # Render index.html anyway to allow frontend to load
    return render_template("index_ver2.html")

# Route to handle chat queries
@app.route("/chat", methods=["POST"])
def chat():
    global access_token, token_expiry
    # Ensure access token is initialized
    if not access_token or datetime.now() >= token_expiry:
        if not authenticate_superset():
            return jsonify({"error": "Failed to authenticate with Superset!"}), 500
    
    output = {}
    output["user_query"] = request.json.get("query")
    if not output["user_query"]:
        output["error"] = "No query provided!"
        return jsonify(output), 400
    
    # Generate SQL using Mistral via Ollama
    output["sql_query"], output["explanation"], output["payload"] = query_ollama(output["user_query"])
    if not output["sql_query"]:
        output["error"] = "Failed to generate SQL query!"
        return jsonify(output), 500
    
    # Execute SQL query in Superset
    output["results"], output['error'] = execute_superset_query(output["sql_query"])
    #create_superset_chart(sql_query=output["sql_query"])
    if not output["results"]:
        output["error"] += " Failed to execute query in Superset!"
        return jsonify(output), 500
    
    return jsonify(output)


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
    parser = argparse.ArgumentParser(description="Run the Superset-Ollama Flask app.")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the Flask app on")
    parser.add_argument("--model", type=str, default=model, help="Ollama model to use")
    args = parser.parse_args()

    # Update model if provided and valid
    if args.model:
        # Check if the model exists in the Ollama library
        try:
            ollama_models_resp = requests.get(MODEL_URL)
            if ollama_models_resp.status_code == 200:
                available_models = [m["name"].replace(":latest", "") for m in ollama_models_resp.json().get("models", [])]
                if args.model in available_models:
                    model = args.model
                else:
                    logger.warning(f"Using default model '{model}' as given Model-arg:'{args.model}' not found in local Ollama-models: {available_models}.")
            else:
                logger.warning(f"Could not fetch Ollama model list (status {ollama_models_resp.status_code}). Using default model '{model}'.")
        except Exception as e:
            logger.warning(f"Error checking Ollama models: {e}. Using default model '{model}'.")

    logger.info(f"Starting Flask app with model: {model} on port {args.port}")
    app.run(debug=True, port=args.port)