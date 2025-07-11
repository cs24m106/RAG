import os, sys
curr_path = os.path.abspath(__file__)
REPO_DIR = "RAG"
REPO_PATH = curr_path[:curr_path.find(REPO_DIR) + len(REPO_DIR)]

ROOT_DIR = REPO_DIR + "_api"
ROOT_PATH = os.path.join(REPO_PATH, "ref/Telco-RAG/Telco-RAG_api")
sys.path.append(ROOT_PATH)
import preheader, logging # import for custom logger
logger = logging.getLogger(__name__) # Setup logging

from analysis import create_pg_engine, load_pm_table_mapping, get_pm_value, make_timezone_unaware
from degraded_kpi import analyze_degraded_kpi_event, save_degraded_event_to_excel
from sqlalchemy import Table, MetaData, select
from datetime import datetime, timedelta, timezone
import pandas as pd
import math

def save_healthy_events_to_excel(healthy_records, filename="output/healthy_kpis.xlsx"):
    if not healthy_records:
        logger.info("No healthy records to save.")
        return

    rows = []
    for idx, record in enumerate(healthy_records):
        # Safely extract base fields
        timestamp = record.get("timestamp")
        cell_id = record.get("cell_id")
        kpi = record.get("target_kpi")
        if not cell_id or not kpi:
            logger.warning(f"Skipping record {idx} due to missing required field(s): "
                           f"cell_id={cell_id}, kpi={kpi}")
            continue

        base_row = {
            "time": timestamp,
            "cell_id": cell_id,
            "kpi": kpi
        }

        pms = record.get("pm_values", {})
        row = base_row.copy()
        for pm, value in pms.items():
            safe_pm_name = pm.replace('.', '_')  # Avoid naming conflict
            row[f"{safe_pm_name}_value"] = value

        rows.append(row)

    if not rows:
        logger.info("No valid healthy records to save after filtering out bad ones.")
        return

    df = pd.DataFrame(rows)
    # Build final column list
    cols = ["time", "cell_id", "kpi"]
    added = set(cols)
    for col in df.columns:
        if "_value" in str(col) and col not in added:
            cols.append(col)
            added.add(col)

    df = df[cols]
    df = make_timezone_unaware(df)

    # --- Compute average values ---
    avg_row = {"time": "AVG", "cell_id": "", "kpi": ""}
    value_cols = [col for col in df.columns if "_value" in col]
    for col in value_cols:
        avg_value = df[col].mean()
        avg_row[col] = avg_value

    # --- Compute median values ---
    median_row = {"time": "MEDIAN", "cell_id": "", "kpi": ""}

    for col in value_cols:
        median_value = df[col].median()
        median_row[col] = median_value

    # Convert to DataFrames
    df_avg = pd.DataFrame([avg_row])
    df_median = pd.DataFrame([median_row])
    # Combine original data + stats rows
    df_final = pd.concat([df, df_avg, df_median], ignore_index=True)

    # --- Save to Excel ---
    df_final.to_excel(filename, index=False)
    logger.info(f"Healthy KPI records saved to '{filename}' including AVG and MEDIAN rows")

# Helper Function: Get healthy PM records from past for comparison
def get_healthy_pm_records(target_kpi, cell_id, dependent_pms, hours=24, threshold=90.0, engine=None):
    if not engine:
        raise Exception("PostgreSQL engine must be provided")
    
    logger.info(f"Fetching healthy PM records for {target_kpi} on cell {cell_id}")

    # Step 1: Load PM mapping
    pm_to_table_map = load_pm_table_mapping()

    # Step 2: Find healthy KPI times
    try:
        metadata = MetaData()
        kpi_table = Table('hrly_kpi_1', metadata, autoload_with=engine)

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)

        stmt = (
            select(kpi_table.c.time)
            .where(
                kpi_table.c.cell_id == cell_id,
                kpi_table.c[target_kpi].is_not(None),
                kpi_table.c[target_kpi] > threshold,
                kpi_table.c.time >= start_time,
                kpi_table.c.time <= end_time
            )
            .order_by(kpi_table.c.time.desc())
        )

        df = pd.read_sql_query(stmt, engine.connect())
        healthy_times = df['time'].tolist()
        logger.info(f"Found {len(healthy_times)} healthy timestamps for KPI '{target_kpi}'")

        if not healthy_times:
            return []

        # Step 3: For each healthy time, fetch all PMs
        healthy_records = []
        for timestamp in healthy_times:
            record = {
                'timestamp': timestamp,
                'cell_id': cell_id,           
                'target_kpi': target_kpi,    
                'pm_values': {}
            }

            for pm in dependent_pms:
                value = get_pm_value(pm, cell_id, timestamp, pm_to_table_map, engine=engine)
                record['pm_values'][pm] = value

            healthy_records.append(record)

        return healthy_records

    except Exception as e:
        logger.error(f"Error fetching healthy PM records: {e}")
        return []
    

def compare_degraded_with_healthy(degraded_event, healthy_records):
    """
    Compares the PM values from a degraded KPI event with the average and median PM values
    from healthy records, using standardized Z-scores for comparison.
    """

    if not healthy_records:
        logger.info("No healthy records found for comparison.")
        return

    # Step 1: Extract all PM names
    pm_values_degraded = degraded_event.get("dependent_pms", {})
    if not pm_values_degraded:
        logger.info("No dependent PMs found in degraded event.")
        return

    # Step 2: Build mean, median, and standard deviation from healthy records
    pm_values_list = {}
    for record in healthy_records:
        pms = record.get("pm_values", {})
        for pm_name, value in pms.items():
            if isinstance(value, (int, float)):
                if pm_name not in pm_values_list:
                    pm_values_list[pm_name] = []
                pm_values_list[pm_name].append(value)

    pm_mean = {pm: sum(vals) / len(vals) for pm, vals in pm_values_list.items()}
    pm_median = {pm: sorted(vals)[len(vals) // 2] for pm, vals in pm_values_list.items()}  # Simple median
    pm_std = {pm: (sum((x - pm_mean[pm]) ** 2 for x in vals) / len(vals)) ** 0.5 for pm, vals in pm_values_list.items()}  # Sample std dev

    # Step 3: Compare and display results
    logger.info("\n=== Comparison: Degraded vs Healthy Baseline (Standardized) ===")
    logger.info(f"{'PM Name':<40} | {'Degraded (Z)':>13} | {'Healthy Avg (Z)':>15} | {'Median (Z)':>13} | {'Diff (Avg Z)':>14} | {'Diff (Median Z)':>15}")

    for pm_name in sorted(pm_values_degraded.keys()):
        degraded_value = pm_values_degraded.get(pm_name)
        healthy_mean = pm_mean.get(pm_name)
        healthy_median = pm_median.get(pm_name)
        healthy_std = pm_std.get(pm_name)

        # Skip if all data is missing
        if degraded_value is None and healthy_mean is None and healthy_median is None:
            logger.info(f"{pm_name:<40} | {'None':>13} | {'None':>15} | {'None':>13} | {'N/A':>14} | {'N/A':>15}")
            continue

        # Standardize values
        degraded_z = (degraded_value - healthy_mean) / healthy_std if healthy_std and healthy_std != 0 else float('nan')
        healthy_mean_z = 0.0 if healthy_std else float('nan')  # Mean is 0 in Z-score scale
        healthy_median_z = (healthy_median - healthy_mean) / healthy_std if healthy_std and healthy_std != 0 else float('nan')

        # Calculate differences in Z-scores
        diff_avg_z = degraded_z - healthy_mean_z if not math.isnan(degraded_z) else float('nan')
        diff_median_z = degraded_z - healthy_median_z if not math.isnan(degraded_z) and not math.isnan(healthy_median_z) else float('nan')

        # Print based on available data
        msg = f"{pm_name:<40} | "

        if not math.isnan(degraded_z):
            msg += f"{degraded_z:>13.2f} | "
        else:
            msg += f"{'None':>13} | "

        if not math.isnan(healthy_mean_z):
            msg += f"{healthy_mean_z:>15.2f} | "
        else:
            msg += f"{'None':>15} | "

        if not math.isnan(healthy_median_z):
            msg += f"{healthy_median_z:>13.2f} | "
        else:
            msg += f"{'None':>13} | "

        if not math.isnan(diff_avg_z):
            msg += f"{diff_avg_z:>14.2f} | "
        else:
            msg += f"{'N/A':>14} | "

        if not math.isnan(diff_median_z):
            msg += f"{diff_median_z:>15.2f}"
        else:
            msg += f"{'N/A':>15}"

        logger.info(msg)


# Main execution
if __name__ == "__main__":
    engine = create_pg_engine()
    if not engine:
        logger.info("Error: Failed to create PostgreSQL engine")
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
                logger.info("Invalid numeric input for hours or threshold")
                continue

            try:
                # Analyze degraded event
                result = analyze_degraded_kpi_event(target_kpi, cell_id, hours=hours, threshold=threshold, engine=engine)
                logger.info("\n=== Degraded KPI Event Analysis ===")
                logger.info(f"Target KPI: {result['target_kpi']}")
                logger.info(f"Cell ID: {result['cell_id']}")
                logger.info(f"Timestamp: {result['timestamp']}")
                logger.info(f"KPI Value: {result['kpi_value']}")

                logger.info("\nTop 3 Correlated KPIs:")
                for idx, kpi in enumerate(result['top_kpis'], 1):
                    logger.debug(f"  {idx}. {kpi}")

                logger.info("\nDependent PM Values:")
                for pm, val in sorted(result['dependent_pms'].items()):
                    logger.debug(f"  {pm}: {val if val is not None else 'No data'}")

                # Save degraded event
                save_degraded_event_to_excel(result)

                # Get and save healthy records
                logger.info("\nFetching healthy baseline PM records...")
                healthy_records = get_healthy_pm_records(
                    target_kpi=result['target_kpi'],
                    cell_id=result['cell_id'],
                    dependent_pms=list(result['dependent_pms'].keys()),
                    hours=24,
                    threshold=90.0,
                    engine=engine
                )
                save_healthy_events_to_excel(healthy_records)

                # Compare degraded with healthy baseline
                compare_degraded_with_healthy(result, healthy_records)

            except Exception as e:
                logger.error(f"Error: {e}")

    finally:
        engine.dispose()