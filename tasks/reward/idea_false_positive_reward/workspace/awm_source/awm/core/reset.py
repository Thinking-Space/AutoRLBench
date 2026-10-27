import json
import os
from loguru import logger
from dataclasses import dataclass
from awm.core.db import create_sqlite_database
from awm.core.sample import execute_sample_data
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from awm.tools import tools_jsonl_load, normalize_scenario_name, find_scenario_entry

@dataclass
class Config:
    input_db: str # database schema file
    input_sample: str # sample data file
    database_dir: str = './outputs/databases'
    scenarios: list[str] | None = None

    def pre_process(self):
        assert os.path.exists(self.input_db), f"Database schema file {self.input_db} not found"
        assert os.path.exists(self.input_sample), f"Sample data file {self.input_sample} not found"
        if self.scenarios:
            self.scenarios = {normalize_scenario_name(name) for name in self.scenarios}

def process_schema(schema_item, database_dir, sample_data_dict):
    scenario_name = schema_item["scenario"]
    db_schema = schema_item["db_schema"]

    logger.info(f"resetting database for {scenario_name}")

    db_path, _, _, _ = create_sqlite_database(scenario_name, db_schema, database_dir)

    successful = 0
    failed = 0
    has_error = False

    if scenario_name in sample_data_dict:
        sample_data = sample_data_dict[scenario_name]
        try:
            successful, failed, _ = execute_sample_data(db_path, sample_data, scenario_name)
        except Exception as e:
            logger.error(f"Error executing sample data for {scenario_name}: {e}")
            has_error = True

        if failed > 0:
            has_error = True
    else:
        has_error = True
        logger.error(f"No sample data found for {scenario_name}")

    return scenario_name, successful, failed, has_error


def reset_single_database(
    input_db: str,
    input_sample: str,
    scenario: str,
    database_dir: str = './outputs/databases',
) -> str:
    """Reset the database for a single scenario.

    Loads the schema and sample data from JSONL files, creates the SQLite
    database, and populates it.

    Args:
        input_db: path to gen_db.jsonl
        input_sample: path to gen_sample.jsonl
        scenario: scenario name (will be normalized)
        database_dir: directory to write the .db file into

    Returns:
        The path to the created database file.

    Raises:
        ValueError: if the scenario is not found in the data files.
        RuntimeError: if the database creation fails.
    """
    scenario_norm = normalize_scenario_name(scenario)
    os.makedirs(database_dir, exist_ok=True)

    db_schemas_data = tools_jsonl_load(input_db)
    schema_item = find_scenario_entry(db_schemas_data, scenario_norm)
    if not schema_item:
        raise ValueError(f"Scenario '{scenario}' not found in {input_db}")

    sample_data_list = tools_jsonl_load(input_sample)
    sample_item = find_scenario_entry(sample_data_list, scenario_norm)
    if not sample_item:
        raise ValueError(f"Scenario '{scenario}' not found in {input_sample}")

    sample_data_dict = {sample_item["scenario"]: sample_item["sample_data"]}
    _, successful, failed, has_error = process_schema(schema_item, database_dir, sample_data_dict)

    db_path = os.path.join(database_dir, f"{scenario_norm}.db")
    if not os.path.exists(db_path):
        raise RuntimeError(f"Database file not created: {db_path}")

    if has_error:
        logger.warning(f"Database for {scenario} created with errors (successful={successful}, failed={failed})")

    return db_path

def reset_all_databases(
    input_db: str,
    input_sample: str,
    database_dir: str = './outputs/databases',
    scenarios: set[str] | None = None,
):


    logger.info("="*100)
    logger.info("starting database reset")
    logger.info("="*100)
    
    db_schemas_data = tools_jsonl_load(input_db)
    sample_data_list = tools_jsonl_load(input_sample)

    if scenarios:
        db_schemas_data = [item for item in db_schemas_data if normalize_scenario_name(item["scenario"]) in scenarios]
        sample_data_list = [item for item in sample_data_list if normalize_scenario_name(item["scenario"]) in scenarios]
        logger.info(f"Resetting databases for {len(scenarios)} scenarios: {scenarios}")
    
    sample_data_dict = {item["scenario"]: item["sample_data"] for item in sample_data_list}
    
    total_successful = 0
    total_failed = 0
    no_error_scenarios = []
    error_scenarios = []
    
    with ProcessPoolExecutor(max_workers=min(os.cpu_count() or 4, len(db_schemas_data), 64)) as executor:
        futures = [executor.submit(process_schema, schema_item, database_dir, sample_data_dict) for schema_item in db_schemas_data]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Resetting databases"):
            scenario_name, successful, failed, has_error = future.result()
            total_successful += successful
            total_failed += failed
            if has_error:
                error_scenarios.append(scenario_name)
            else:
                no_error_scenarios.append(scenario_name)
    
    
    logger.info(f"="*100)
    logger.info(f"Database reset done! Total/Success/Has_Errors = {len(db_schemas_data)}/{total_successful}/{total_failed}")
    logger.info(f"Scenarios error free: {len(no_error_scenarios)}\n{no_error_scenarios}")
    logger.info(f"Scenarios with errors: {len(error_scenarios)}\n{error_scenarios}")


def run(config: Config):
    reset_all_databases(config.input_db, config.input_sample, config.database_dir, config.scenarios)
