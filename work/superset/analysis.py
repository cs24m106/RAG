import os
import json, yaml
from datetime import datetime, timedelta
import pandas as pd
import requests, logging
logger = logging.getLogger(__name__) # Setup logging (existing)

from sqlalchemy import create_engine, Table, MetaData, select

# Configuration for Superset (existing)
SUPERSET_URL = "http://10.100.80.31:8088"
DATABASE_ID = 2
SUPERSET_USERNAME = os.getenv("SUPERSET_USERNAME", "admin")
SUPERSET_PASSWORD = os.getenv("SUPERSET_PASSWORD", "admin")

# Configuration for PostgreSQL
PG_HOST = os.getenv("PG_HOST", "10.100.80.23")  # Adjust as needed
PG_PORT = os.getenv("PG_PORT", "5532")
PG_DATABASE = os.getenv("PG_DATABASE", "analytics")  # Replace with actual DB name
PG_USERNAME = os.getenv("PG_USERNAME", "db-admin")  # Replace with actual username
PG_PASSWORD = os.getenv("PG_PASSWORD", "db-admin")  # Replace with actual password


# --------------------------------------------------------------------------------------------------------
#                                           Superset Handlers
# --------------------------------------------------------------------------------------------------------

# Token storage (existing)
access_token = None
refresh_token = None
token_expiry = None
session = requests.Session()

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
            logger.info("Successfully authenticated with Superset")
            logger.info(f"Session cookies: {session.cookies.get_dict()}")
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
            logger.info("Successfully refreshed access token")
            logger.info(f"Session cookies: {session.cookies.get_dict()}")
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
        logger.info("Access token missing or expired, refreshing")
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
            logger.info("Access token invalid, refreshing")
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
        "queryLimit": 10000000  # Increase limit if needed
    }
    
    try:
        response = session.post(f"{SUPERSET_URL}/api/v1/sqllab/execute/", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json().get("data", [])
        elif response.status_code == 401:
            logger.info("Access token invalid, refreshing")
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

# --------------------------------------------------------------------------------------------------------
#                                           Postgress Handlers
# --------------------------------------------------------------------------------------------------------
datapath = "/home/techie/logeshv/RAG/work/chromadb/documents/"

# Helper function to create PostgreSQL connection
def create_pg_engine():
    try:
        connection_string = f"postgresql://{PG_USERNAME}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
        engine = create_engine(connection_string)
        logger.info("Successfully created PostgreSQL engine")
        return engine
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL engine: {e}")
        return None

# Helper function to execute SQL query directly on PostgreSQL
def execute_pg_query(sql_query):
    engine = create_pg_engine()
    if not engine:
        logger.error("No PostgreSQL engine available")
        return None
    try:
        df = pd.read_sql_query(sql_query, engine)
        logger.info(f"Fetched {len(df)} rows from PostgreSQL")
        return df
    except Exception as e:
        logger.error(f"PostgreSQL query error: {e}")
        return None
    finally:
        engine.dispose()  # Clean up connection

# Function 1: Get top 3 correlated KPIs from AllMetrics.xlsx
def get_top_n_correlated_kpis(target_metric, file_path=os.path.join(datapath,"AllMetrics.xlsx"), top_n=3):
    if top_n < 1 or top_n > 10:
        logger.warning("top_n must be between 1 and 10")
    if top_n < 1: top_n = 1;
    if top_n > 10: top_n = 10;

    df = pd.read_excel(file_path)
    row = df[df['Target Metric'] == target_metric]
    if row.empty:
        logger.error(f"Target metric '{target_metric}' not found in {file_path}")
        raise ValueError

    top_kpis = []
    for i in range(1, top_n + 1):
        col_name = f'Top_{i}_Correlated'
        if col_name not in df.columns:
            logger.error(f"Column '{col_name}' not found in {file_path}")
            raise KeyError
        top_kpis.append(row[col_name].values[0])
    return top_kpis

# Function 2: Get formula for a given KPI from KPI_Formula
def get_formula_for_kpi(kpi_name, file_path=os.path.join(datapath,"KPI_formula.xlsx")):
    df = pd.read_excel(file_path)
    row = df[df['db_column_bame'] == kpi_name]
    if row.empty:
        logger.error(f"KPI '{kpi_name}' not found in {file_path}")
        raise ValueError

    formula_json_str = row['Formula json'].iloc[0]
    if pd.isna(formula_json_str) or formula_json_str.strip() == '':
        return None
    try:
        return json.loads(formula_json_str)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in formula for '{kpi_name}'")
        raise ValueError

# Function 3: Extract all PMs from formula JSON recursively
def extract_pms_from_formula(formula_json):
    logger.info(f"extractings pms out of given formula json: {formula_json}")
    pms = set()
    def recursive_extract(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "$number":
                    if isinstance(value, str) and not value.isdigit(): # adding only variables, not constants
                        if '.' in value or '_' in value or value.startswith("RRU") or value.startswith("ERAB"):
                            pms.add(value)
                else:
                    recursive_extract(value)
        elif isinstance(node, list):
            for item in node:
                recursive_extract(item)
    if formula_json:
        recursive_extract(formula_json)
    return list(pms)

# Function 4: Load PM-to-table mapping from YAML
def load_pm_table_mapping(config_path=os.path.join(datapath,"hourly_pm_ingestion_config.yml")):
    pm_to_table = {}
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if 'hourly-pm-db-ingestion-config' not in config:
            logger.error("'hourly-pm-db-ingestion-config' not found in YAML")
            exit(1)
        tables = config['hourly-pm-db-ingestion-config'].get('tables', [])
        for table in tables:
            table_name = table.get('name')
            if not table_name:
                logger.warning("Table missing 'name' field")
                continue
            columns = table.get('columns', [])
            for column in columns:
                pm_name = column.get('pm-metric-name')
                if not pm_name:
                    logger.warning("Column missing 'pm-metric-name'")
                    continue
                pm_to_table[pm_name] = table_name
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML format in {config_path}: {e}")
    return pm_to_table

# Function 5: Get PM value from the appropriate table
def get_pm_value(pm_name, cell_id, timestamp, pm_to_table_map, engine=None):
    if not engine:
        logger.error("PostgreSQL engine must be provided")
        raise Exception

    # Convert PM name to lowercase with underscores
    pm_name_converted = pm_name.replace('.', '_').lower()
    
    table_name = pm_to_table_map.get(pm_name_converted)
    if not table_name:
        logger.warning(f"PM '{pm_name_converted}' not found in configuration")
        return None

    # Try both original and lowercase table name
    table_names = [table_name.lower()]
    
    for tbl_name in table_names:
        logger.info(f"Attempting to access table '{tbl_name}' for PM '{pm_name_converted}'")
        try:
            metadata = MetaData()
            pm_table = Table(tbl_name, metadata, autoload_with=engine)
            logger.info(f"Successfully loaded table '{tbl_name}'")

            stmt = (
                select(pm_table.c[pm_name_converted])
                .where(
                    pm_table.c.cell_id == cell_id,
                    pm_table.c.time == timestamp
                )
                .limit(1)
            )

            logger.info(f"SQL Query for PM '{pm_name_converted}': \n{str(stmt)}")

            df = pd.read_sql_query(stmt, engine.connect())
            if df.empty or df[pm_name_converted].isna().all():
                logger.warning(f"No data found for PM '{pm_name_converted}' at timestamp {timestamp}")
                return None
            logger.info(f"Data retrieved for PM '{pm_name_converted}': {df[pm_name_converted].iloc[0]}")
            return df[pm_name_converted].iloc[0]

        except KeyError:
            logger.warning(f"PM '{pm_name_converted}' not found as a column in {tbl_name}")
            continue
        except Exception as e:
            logger.warning(f"Error fetching PM '{pm_name_converted}' from table '{tbl_name}': {str(e)}")
            continue
    
    logger.warning(f"Failed to fetch PM '{pm_name_converted}' from any table")
    return None

def make_timezone_unaware(df):
    """Convert all datetime columns to timezone-unaware."""
    for col in df.select_dtypes(include=['datetime64[ns, UTC]']).columns:
        df[col] = df[col].dt.tz_localize(None)
    return df