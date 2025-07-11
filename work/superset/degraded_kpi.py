import os, sys
curr_path = os.path.abspath(__file__)
REPO_DIR = "RAG"
REPO_PATH = curr_path[:curr_path.find(REPO_DIR) + len(REPO_DIR)]

ROOT_DIR = REPO_DIR + "_api"
ROOT_PATH = os.path.join(REPO_PATH, "ref/Telco-RAG/Telco-RAG_api")
sys.path.append(ROOT_PATH)
import preheader, logging # import for custom logger
logger = logging.getLogger(__name__) # Setup logging

from analysis import create_pg_engine, get_top_n_correlated_kpis, get_formula_for_kpi, extract_pms_from_formula, load_pm_table_mapping, get_pm_value, make_timezone_unaware
from sqlalchemy import Table, MetaData, select
from datetime import datetime, timedelta, timezone
import pandas as pd

def save_degraded_event_to_excel(degraded_data, filename="output/degraded_kpi.xlsx"):
    if not degraded_data:
        print("No degraded event data to save.")
        return

    row = {
        "time": degraded_data["timestamp"],
        "cell_id": degraded_data["cell_id"],
        "kpi": degraded_data["target_kpi"],
        "kpi_value": degraded_data["kpi_value"]
    }

    for pm, value in degraded_data["dependent_pms"].items():
        row[pm] = pm
        row[f"{pm}_value"] = value

    df = pd.DataFrame([row])

    cols = ["time", "cell_id", "kpi", "kpi_value"]
    for pm in degraded_data["dependent_pms"]:
        cols.append(pm)
        cols.append(f"{pm}_value")

    df = df[cols]
    df = make_timezone_unaware(df)

    df.to_excel(filename, index=False)
    print(f"Degraded KPI event saved to '{filename}'")

# Helper Function: Get degraded KPI event from hrly_kpi_1
def get_low_kpi_event(target_kpi, cell_id, hours=24, threshold=20.0, engine=None):
    if not engine:
        logger.error("PostgreSQL engine must be provided")
        exit(1)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours) # taking past 24 hours time-stamp range

    try:
        metadata = MetaData()
        kpi_table = Table('hrly_kpi_1', metadata, autoload_with=engine)

        # Build safe query using SQLAlchemy Core
        stmt = (
            select(kpi_table.c.time, kpi_table.c.cell_id, kpi_table.c[target_kpi].label("kpi_value"))
            .where(
                kpi_table.c.cell_id == cell_id,
                kpi_table.c[target_kpi].is_not(None),
                kpi_table.c[target_kpi] < threshold,
                kpi_table.c.time >= start_time,
                kpi_table.c.time <= end_time
            )
            .order_by(kpi_table.c.time.desc())
            .limit(1)
        )

        df = pd.read_sql_query(stmt, engine.connect())
        if df.empty:
            return None
        return df.iloc[0].to_dict()

    except Exception as e:
        logger.error(f"Error fetching KPI data: {e}")
        exit(1)

# Main analysis function
def analyze_degraded_kpi_event(target_kpi, cell_id, hours=24, threshold=20.0, top_n=3, engine=None):
    # Step 1: Get top 3 correlated KPIs and their dependent PMs
    top_kpis_result = get_top_n_correlated_kpis(target_kpi, top_n=top_n)
    dependent_pms = set()
    for kpi in top_kpis_result:
        formula_json = get_formula_for_kpi(kpi)
        pms = extract_pms_from_formula(formula_json)
        dependent_pms.update(pms)

    # Step 2: Load PM-to-table mapping
    pm_to_table_map = load_pm_table_mapping()

    # Step 3: Find degraded KPI event
    kpi_data = get_low_kpi_event(target_kpi, cell_id, hours=hours, threshold=threshold, engine=engine)
    if not kpi_data:
        logger.error(f"No low-value event found for KPI '{target_kpi}' on cell_id '{cell_id}' in last {hours} hrs")
        exit(1)

    timestamp = kpi_data["time"]
    kpi_value = kpi_data["kpi_value"]

    # Step 4: Fetch values of all dependent PMs
    pm_values = {}
    for pm in dependent_pms:
        value = get_pm_value(pm, cell_id, timestamp, pm_to_table_map, engine=engine)
        pm_values[pm] = value

    return {
        "target_kpi": target_kpi,
        "cell_id": cell_id,
        "timestamp": timestamp,
        "kpi_value": kpi_value,
        "top_kpis": top_kpis_result,
        "dependent_pms": pm_values
    }

# modify db vars here
cell_ids = ['24419330']
target_kpis = ['rrc_connection_success_rate_all']
no_of_hours = 24
threshold = 20

# Main execution
if __name__ == "__main__":

    # Create a single PostgreSQL engine
    engine = create_pg_engine()
    if not engine:
        logger.error("Error: Failed to create PostgreSQL engine")
        exit(1)

    try:
        for cell_id in cell_ids:
            for target_kpi in target_kpis:
                logger.info(f"Cell ID: {cell_id}, Target KPI: {target_kpi}")
                try:
                    result = analyze_degraded_kpi_event(target_kpi, cell_id, hours=no_of_hours, threshold=threshold, engine=engine)
                    logger.info("\n=== Degraded KPI Event Analysis ===")
                    logger.info(f"Target KPI: {result['target_kpi']}")
                    logger.info(f"Cell ID: {result['cell_id']}")
                    logger.info(f"Timestamp: {result['timestamp']}")
                    logger.info(f"KPI Value: {result['kpi_value']}")
                    logger.info("Top 3 Correlated KPIs:")
                    for idx, kpi in enumerate(result['top_kpis'], 1):
                        logger.debug(f"  {idx}. {kpi}")
                    logger.info("Dependent PM Values:")
                    for pm, val in sorted(result['dependent_pms'].items()):
                        logger.info(f"  {pm}: {val if val is not None else 'No data'}")
                except Exception as e:
                    logger.error(f"Error: {e}")
                input("PAUSED! Press <ENTER> to continue...")
                logger.debug("\n")
    finally:
        engine.dispose()