import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
import os


import os
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine

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

# Token storage (existing)
access_token = None
refresh_token = None
token_expiry = None
session = requests.Session()

# Setup logging (existing)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Helper function to create PostgreSQL connection
def create_pg_engine():
    try:
        connection_string = f"postgresql://{PG_USERNAME}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
        engine = create_engine(connection_string)
        logger.debug("Successfully created PostgreSQL engine")
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from two YAML files
def load_config(tables_path: str, columns_path: str) -> tuple[List[str], Dict[str, Dict[str, str]]]:
    try:
        # Load tables
        with open(tables_path, "r") as file:
            tables_config = yaml.safe_load(file)
            tables = tables_config.get("tables", [])
        # Load columns
        with open(columns_path, "r") as file:
            columns_config = yaml.safe_load(file)
            columns = columns_config.get("columns", {})
        return tables, columns
    except Exception as e:
        logger.error(f"Error loading config files: {e}")
        return [], {}

# Build SQL query
def build_sql_query(table_names: List[str], hours: int = 24) -> str:
    if not table_names:
        raise ValueError("No tables provided.")
    base_table = table_names[0]
    base_subquery = f"""
        SELECT *
        FROM {base_table}
        WHERE "time" >= NOW() - INTERVAL '{hours} HOURS'
    """
    sql = f"SELECT * FROM ({base_subquery}) AS {base_table.split('.')[-1]}"
    for table in table_names[1:]:
        alias = table.split('.')[-1]
        subquery = f"""
            SELECT *
            FROM {table}
            WHERE "time" >= NOW() - INTERVAL '{hours} HOURS'
        """
        sql += f"\nLEFT JOIN ({subquery}) AS {alias} ON {base_table.split('.')[-1]}.time = {alias}.time AND {base_table.split('.')[-1]}.cell_id = {alias}.cell_id"
    print(f"Generated SQL Query:\n{sql}\n")
    return sql

# Sample by percentage
def sample_by_percentage(df: pd.DataFrame, metric_col: str, sample_size: int = None) -> pd.DataFrame:
    df = df.copy()
    bins = [i*10 for i in range(11)]
    bins[-1] += 0.01
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    df['bin'] = pd.cut(df[metric_col], bins=bins, include_lowest=True, right=False, labels=labels)
    binned_dfs = []
    for label in labels:
        bin_df = df[df['bin'] == label]
        if not bin_df.empty:
            binned_dfs.append(bin_df)
    sizes = [len(b) for b in binned_dfs]
    print("\nBin Distribution:")
    for label, size in zip(labels, sizes):
        print(f"Bin {label}: {size} entries")
    min_sample_size = min(sizes) if sample_size is None else min(sample_size, min(sizes))
    print(f"\nSampling {min_sample_size} entries from each populated bin.\n")
    sampled = []
    for bdf in binned_dfs:
        sampled.append(bdf.sample(n=min_sample_size, random_state=42))
    return pd.concat(sampled).drop(columns=['bin'])

# Sample by count
def sample_by_count(df: pd.DataFrame, metric_col: str, sample_size: int = None) -> pd.DataFrame:
    df = df.copy()
    bins = pd.qcut(df[metric_col], q=10, duplicates='drop').cat.categories
    if len(bins) < 2:
        return df.sample(n=sample_size or len(df), random_state=42)
    df['bin'] = pd.cut(df[metric_col], bins=bins, include_lowest=True)
    binned_dfs = [df[df['bin'] == label] for label in df['bin'].cat.categories]
    sizes = [len(b) for b in binned_dfs]
    print("\nBin Distribution:")
    for label, size in zip(df['bin'].cat.categories, sizes):
        print(f"Bin {label}: {size} entries")
    min_sample_size = min(sizes) if sample_size is None else min(sample_size, min(sizes))
    print(f"\nSampling {min_sample_size} entries from each populated bin.\n")
    sampled = [bdf.sample(n=min_sample_size, random_state=42) for bdf in binned_dfs if not bdf.empty]
    return pd.concat(sampled).drop(columns=['bin'])

# Sample by volume
def sample_by_volume(df: pd.DataFrame, metric_col: str, sample_size: int = None) -> pd.DataFrame:
    df = df.copy()
    df['log_metric'] = np.log1p(df[metric_col])
    bins = pd.qcut(df['log_metric'], q=10, duplicates='drop').cat.categories
    if len(bins) < 2:
        return df.sample(n=sample_size or len(df), random_state=42)
    df['bin'] = pd.cut(df['log_metric'], bins=bins, include_lowest=True)
    binned_dfs = [df[df['bin'] == label] for label in df['bin'].cat.categories]
    sizes = [len(b) for b in binned_dfs]
    print("\nBin Distribution:")
    for label, size in zip(df['bin'].cat.categories, sizes):
        print(f"Bin {label}: {size} entries")
    min_sample_size = min(sizes) if sample_size is None else min(sample_size, min(sizes))
    print(f"\nSampling {min_sample_size} entries from each populated bin.\n")
    sampled = [bdf.sample(n=min_sample_size, random_state=42) for bdf in binned_dfs if not bdf.empty]
    return pd.concat(sampled).drop(columns=['bin', 'log_metric'])

# Sample by rate
def sample_by_rate(df: pd.DataFrame, metric_col: str, sample_size: int = None) -> pd.DataFrame:
    df = df.copy()
    bins = pd.qcut(df[metric_col], q=10, duplicates='drop').cat.categories
    if len(bins) < 2:
        return df.sample(n=sample_size or len(df), random_state=42)
    df['bin'] = pd.cut(df[metric_col], bins=bins, include_lowest=True)
    binned_dfs = [df[df['bin'] == label] for label in df['bin'].cat.categories]
    sizes = [len(b) for b in binned_dfs]
    print("\nBin Distribution:")
    for label, size in zip(df['bin'].cat.categories, sizes):
        print(f"Bin {label}: {size} entries")
    min_sample_size = min(sizes) if sample_size is None else min(sample_size, min(sizes))
    print(f"\nSampling {min_sample_size} entries from each populated bin.\n")
    sampled = [bdf.sample(n=min_sample_size, random_state=42) for bdf in binned_dfs if not bdf.empty]
    return pd.concat(sampled).drop(columns=['bin'])

# Sample by time
def sample_by_time(df: pd.DataFrame, metric_col: str, sample_size: int = None) -> pd.DataFrame:
    df = df.copy()
    df['log_metric'] = np.log1p(df[metric_col])
    bins = pd.qcut(df['log_metric'], q=10, duplicates='drop').cat.categories
    if len(bins) < 2:
        return df.sample(n=sample_size or len(df), random_state=42)
    df['bin'] = pd.cut(df['log_metric'], bins=bins, include_lowest=True)
    binned_dfs = [df[df['bin'] == label] for label in df['bin'].cat.categories]
    sizes = [len(b) for b in binned_dfs]
    print("\nBin Distribution:")
    for label, size in zip(df['bin'].cat.categories, sizes):
        print(f"Bin {label}: {size} entries")
    min_sample_size = min(sizes) if sample_size is None else min(sample_size, min(sizes))
    print(f"\nSampling {min_sample_size} entries from each populated bin.\n")
    sampled = [bdf.sample(n=min_sample_size, random_state=42) for bdf in binned_dfs if not bdf.empty]
    return pd.concat(sampled).drop(columns=['bin', 'log_metric'])


# Sample data
def sample_data(df: pd.DataFrame, metric_col: str, metric_type: str) -> pd.DataFrame:
    if metric_type == "percentage":
        return sample_by_percentage(df, metric_col)
    elif metric_type == "count":
        return sample_by_count(df, metric_col)
    elif metric_type == "volume":
        return sample_by_volume(df, metric_col)
    elif metric_type == "rate":
        return sample_by_rate(df, metric_col)
    elif metric_type == "time":
        return sample_by_time(df, metric_col)
    else:
        raise NotImplementedError(f"Sampling method for metric type '{metric_type}' is not implemented.")

# Calculate correlations 
def calculate_correlations(df: pd.DataFrame, metric_col: str) -> Dict[str, float]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if metric_col not in numeric_cols:
        raise ValueError(f"Metric column '{metric_col}' must be numeric")
    correlations = {}
    for col in numeric_cols:
        if col != metric_col:
            corr = df[[col, metric_col]].corr().iloc[0,1]
            correlations[col] = corr
    return correlations

# Save results
def save_results(correlation_dict: dict, top_n: int = 10, output_file: str = "output/output.txt"):
    with open(output_file, "w") as f:
        f.write("Full Correlation Matrix:\n")
        for col, val in correlation_dict.items():
            f.write(f"{col}: {val:.4f}\n")
        valid_correlations = [(col, val) for col, val in correlation_dict.items() if not np.isnan(val)]
        sorted_corr = sorted(valid_correlations, key=lambda x: abs(x[1]), reverse=True)
        f.write("\nTop {} Correlated Columns:\n".format(top_n))
        for col, val in sorted_corr[:top_n]:
            f.write(f"{col}: {val:.4f}\n")
    logger.info(f"Results saved to {output_file}")

# New function to process all metrics
def process_all_metrics(tables_path: str = "kpi_tables.yml", columns_path: str = "kpi_columns.yml", hours: int = 24, output_file: str = "output/All_metrics.xlsx"):
    logger.info("Starting processing for all metrics...")
    
    # Load configuration
    tables, columns = load_config(tables_path, columns_path)
    if not tables or not columns:
        logger.error("Failed to load configuration.")
        return

    # Filter columns with metric types: percentage, count, volume, rate
    target_metric_types = ["percentage", "count", "volume", "rate"]
    target_columns = {
        col: info for col, info in columns.items()
        if info.get("metric_type") in target_metric_types and info.get("data_type") == "NUMERIC"
    }
    logger.info(f"Found {len(target_columns)} target columns: {list(target_columns.keys())}")

    # Build and execute SQL query
    logger.info(f"Building SQL query with time filter: last {hours} hours...")
    sql_query = build_sql_query(tables, hours)
    logger.info("Fetching data from PostgreSQL...")
    df = execute_pg_query(sql_query)
    if df is None:
        logger.error("Failed to fetch data from PostgreSQL.")
        return

    # Save raw data
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "raw_postgres_data.csv")
    df.to_csv(csv_file, index=False)
    logger.info(f"Raw data saved to {csv_file}")

    # Drop identifier columns
    identifier_cols = [col for col, info in columns.items() if info["metric_type"] == "identifier"]
    columns_to_drop = ['time'] + identifier_cols
    existing_columns = df.columns.tolist()
    to_drop_existing = [col for col in columns_to_drop if col in existing_columns]
    if to_drop_existing:
        df.drop(columns=to_drop_existing, inplace=True)
        logger.info(f"Dropped columns: {to_drop_existing}")
    else:
        logger.warning("No identifier columns dropped.")

    # Initialize Excel output DataFrame
    excel_columns = [
        "Target Metric", "Top_1_Correlated", "Corr_1", "Top_2_Correlated", "Corr_2",
        "Top_3_Correlated", "Corr_3", "Top_4_Correlated", "Corr_4", "Top_5_Correlated", "Corr_5",
        "Top_6_Correlated", "Corr_6", "Top_7_Correlated", "Corr_7", "Top_8_Correlated", "Corr_8",
        "Top_9_Correlated", "Corr_9", "Top_10_Correlated", "Corr_10"
    ]
    excel_data = []

    # Process each target column
    for metric_col, info in target_columns.items():
        metric_type = info.get("metric_type")
        logger.info(f"Processing metric '{metric_col}' (type: {metric_type})...")

        # Sample data based on metric type
        try:
            sampled_df = sample_data(df, metric_col, metric_type)
            logger.info(f"Sampled data for {metric_col} with shape: {sampled_df.shape}")
        except Exception as e:
            logger.error(f"Failed to sample data for {metric_col}: {e}")
            continue

        # Calculate correlations
        try:
            correlations = calculate_correlations(sampled_df, metric_col)
        except ValueError as e:
            logger.error(f"Failed to calculate correlations for {metric_col}: {e}")
            continue

        # Get top 10 correlations
        valid_correlations = [(col, val) for col, val in correlations.items() if not np.isnan(val)]
        sorted_corr = sorted(valid_correlations, key=lambda x: abs(x[1]), reverse=True)[:10]
        
        # Prepare row for Excel
        row = [metric_col]
        for i, (col, val) in enumerate(sorted_corr, 1):
            row.extend([col, round(val, 4)])
        # Pad row with empty strings if fewer than 10 correlations
        while len(row) < len(excel_columns):
            row.extend(["", 0.0])
        excel_data.append(row)

    # Save to Excel
    if excel_data:
        excel_df = pd.DataFrame(excel_data, columns=excel_columns)
        excel_df.to_excel(output_file, index=False)
        logger.info(f"All metrics correlations saved to {output_file}")
    else:
        logger.warning("No data to save to Excel.")


# Main pipeline
def main():
    tables_path = "kpi_tables.yml"
    columns_path = "kpi_columns.yml"
    metric_col = input("Enter the input metric column name: ")
    try:
        hours = int(input("Enter the number of hours to filter data (e.g., 24): "))
    except ValueError:
        logger.error("Invalid input for hours. Using default value of 24.")
        hours = 24

    logger.info("Loading configuration...")
    tables, columns = load_config(tables_path, columns_path)
    if not tables:
        logger.error("No tables found in config.")
        return

    # Validate metric column
    if metric_col not in columns:
        logger.error(f"Metric column '{metric_col}' not found in configuration.")
        return
    metric_type = columns[metric_col].get("metric_type")
    data_type = columns[metric_col].get("data_type")
    if metric_type == "identifier":
        logger.error(f"Metric column '{metric_col}' is an identifier and cannot be used for correlation.")
        return
    if data_type not in ["NUMERIC"]:
        logger.error(f"Metric column '{metric_col}' has data type '{data_type}' and must be NUMERIC for correlation.")
        return
    logger.info(f"Metric '{metric_col}' identified as type: {metric_type}, data type: {data_type}")

    logger.info(f"Building SQL query with time filter: last {hours} hours...")
    sql_query = build_sql_query(tables, hours)

    logger.info("Fetching data from PostgreSQL...")
    df = execute_pg_query(sql_query)  # Assumes execute_pg_query is defined
    if df is None:
        logger.error("Failed to fetch data from PostgreSQL.")
        return

    # Save as CSV
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "raw_postgres_data.csv")
    df.to_csv(csv_file, index=False)
    logger.info(f"Raw data saved to {csv_file}")

    logger.info(f"Fetched data with shape: {df.shape}")

    # Drop identifier columns
    identifier_cols = [col for col, info in columns.items() if info["metric_type"] == "identifier"]
    columns_to_drop = ['time'] + identifier_cols
    existing_columns = df.columns.tolist()
    to_drop_existing = [col for col in columns_to_drop if col in existing_columns]
    if to_drop_existing:
        df.drop(columns=to_drop_existing, inplace=True)
        logger.info(f"Dropped columns: {to_drop_existing}")
    else:
        logger.warning("No identifier columns dropped.")

    logger.info("Sampling data...")
    sampled_df = sample_data(df, metric_col, metric_type)
    logger.info(f"Sampled data with shape: {sampled_df.shape}")

    logger.info("Calculating correlations...")
    try:
        correlations = calculate_correlations(sampled_df, metric_col)
    except ValueError as ve:
        logger.error(str(ve))
        return

    logger.info("Saving results...")
    save_results(correlations)

if __name__ == "__main__":
    # main()
    process_all_metrics(tables_path="kpi_tables.yml", columns_path="kpi_columns.yml", hours=24, output_file="output/AllMetrics.xlsx")