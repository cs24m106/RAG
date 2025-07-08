import os, sys
curr_path = os.path.abspath(__file__)
REPO_DIR = "RAG"
REPO_PATH = curr_path[:curr_path.find(REPO_DIR) + len(REPO_DIR)]

ROOT_DIR = REPO_DIR + "_api"
ROOT_PATH = os.path.join(REPO_PATH, "ref/Telco-RAG/Telco-RAG_api")
sys.path.append(ROOT_PATH)
import preheader, logging # import for custom logger
logger = logging.getLogger(__name__) # Setup logging

from .analysis import create_pg_engine, get_top_n_correlated_kpis, get_formula_for_kpi, extract_pms_from_formula, load_pm_table_mapping, get_low_kpi_event, get_pm_value


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
        raise Exception(f"No low-value event found for KPI '{target_kpi}' on cell_id '{cell_id}' in last {hours} hrs")

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

# Main execution
if __name__ == "__main__":

    # Create a single PostgreSQL engine
    engine = create_pg_engine()
    if not engine:
        print("Error: Failed to create PostgreSQL engine")
        exit(1)

    try:
        while True:
            target_kpi = input("Enter the target KPI (or 'quit' to exit): ").strip()
            if target_kpi.lower() == 'quit':
                break

            cell_id = input("Enter the cell ID: ").strip()
            hours_input = input("How many hours back to look? (default 24): ").strip()
            threshold_input = input("What is the threshold? (default 20): ").strip()

            try:
                hours = int(hours_input) if hours_input else 24
                threshold = float(threshold_input) if threshold_input else 20.0
            except ValueError:
                print("Invalid numeric input for hours or threshold")
                continue

            try:
                result = analyze_degraded_kpi_event(target_kpi, cell_id, hours=hours, threshold=threshold, engine=engine)
                print("\n=== Degraded KPI Event Analysis ===")
                print(f"Target KPI: {result['target_kpi']}")
                print(f"Cell ID: {result['cell_id']}")
                print(f"Timestamp: {result['timestamp']}")
                print(f"KPI Value: {result['kpi_value']}")
                print("\nTop 3 Correlated KPIs:")
                for idx, kpi in enumerate(result['top_kpis'], 1):
                    print(f"  {idx}. {kpi}")
                print("\nDependent PM Values:")
                for pm, val in sorted(result['dependent_pms'].items()):
                    print(f"  {pm}: {val if val is not None else 'No data'}")
            except Exception as e:
                print(f"\nError: {e}")
    finally:
        engine.dispose()