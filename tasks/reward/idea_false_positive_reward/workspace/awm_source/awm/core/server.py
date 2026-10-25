from dataclasses import dataclass
import os
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from loguru import logger

from awm.tools import tools_jsonl_load, normalize_scenario_name
from awm.core.reset import reset_single_database

@dataclass
class Config:
    scenario: str
    envs_load_path: str # specify a path to load generated_envs.jsonl
    db_path: str | None = None # specify a path to load the database file, xxx.db
    host: str = "127.0.0.1"
    port: int = 8001
    temp_server_path: str | None = None # specify a temp server path, the code will be written to this path
    # Path to generated db schemas file (used when db_path is None to create DB from schema)
    db_schema_path: str = "./outputs/gen_db.jsonl"
    # Path to generated sample data file (used when db_path is None)
    sample_path: str = "./outputs/gen_sample.jsonl"
    # Output directory for server artifacts. Auto-created if not specified.
    output_dir: str | None = None

    def pre_process(self):
        self.scenario = normalize_scenario_name(self.scenario)
        assert os.path.exists(self.envs_load_path), f"Environment file {self.envs_load_path} not found"


def format_raw_code_to_lines(raw_code: str, indent: int) -> list[str]:
    no_indent_code = textwrap.dedent(raw_code).strip()
    indent_code = textwrap.indent(no_indent_code, ' ' * indent)
    return indent_code.split("\n")


def _prepare_server_dir(config: Config) -> str:
    """Prepare the server output directory.

    Creates outputs/servers/<timestamp>_<scenario> if output_dir is not set.
    Returns the output directory path.
    """
    if config.output_dir:
        output_dir = config.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("outputs", "servers", f"{timestamp}_{config.scenario}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _prepare_database(config: Config, output_dir: str) -> str:
    """Prepare the database for the server.

    If db_path is specified, uses it directly.
    Otherwise, creates a fresh DB via reset_single_database.
    Saves a copy as initial.db in the output directory.

    Returns the path to the working database file.
    """
    if config.db_path:
        db_file_path = config.db_path
        assert os.path.exists(db_file_path), f"Database file {db_file_path} not found"
        # Save initial DB snapshot
        initial_db_path = os.path.join(output_dir, "initial.db")
        shutil.copy2(db_file_path, initial_db_path)
        logger.info(f"Saved initial database to {initial_db_path}")
    else:
        # Create DB directly in output_dir as final.db (the working copy)
        db_file_path = reset_single_database(
            input_db=config.db_schema_path,
            input_sample=config.sample_path,
            scenario=config.scenario,
            database_dir=output_dir,
        )
        # reset_single_database names it <scenario>.db, rename to final.db
        final_db_path = os.path.join(output_dir, "final.db")
        os.rename(db_file_path, final_db_path)
        db_file_path = final_db_path
        logger.info(f"Created database at {db_file_path}")

        # Save initial DB snapshot (copy of the freshly created DB)
        initial_db_path = os.path.join(output_dir, "initial.db")
        shutil.copy2(db_file_path, initial_db_path)
        logger.info(f"Saved initial database to {initial_db_path}")

    return db_file_path


def _generate_server_code(config: Config, db_path: str) -> str:
    """Generate the MCP server code and write it to a file.

    Returns the path to the generated server code file.
    """
    envs = tools_jsonl_load(config.envs_load_path)
    envs = {normalize_scenario_name(e["scenario"]): e for e in envs}
    env = envs[config.scenario]

    code = env["full_code"]

    # Generated APIs should be safe to retry.  In particular, the mentoring
    # benchmark seeds the requested cohort in the initial database; a model
    # may therefore call create_program even though the row already exists.
    # The generated endpoint historically inserted blindly and surfaced the
    # UNIQUE(cohort_name) violation as an opaque MCP 500.  Make this endpoint
    # idempotent at generation time: return an existing row with the same
    # cohort name before attempting the insert.
    create_program_insert = (
        "    program = Program(cohort_name=body.cohort_name, start_date=start_date_obj, "
        "end_date=end_date_obj, target_audience=body.target_audience, "
        "max_participants=body.max_participants, created_by_user_id=CURRENT_USER_ID, "
        "created_at=now, updated_at=now)\n"
        "    session.add(program)"
    )
    create_program_idempotent = (
        "    existing = session.query(Program).filter(Program.cohort_name == body.cohort_name).first()\n"
        "    if existing is not None:\n"
        "        if (existing.start_date != start_date_obj or existing.end_date != end_date_obj "
        "or existing.target_audience != body.target_audience or existing.max_participants != body.max_participants):\n"
        "            session.close()\n"
        "            raise HTTPException(status_code=409, detail='A program with this cohort_name already exists with different fields')\n"
        "        result = ProgramResponse(id=existing.id, cohort_name=existing.cohort_name, "
        "start_date_value=existing.start_date, end_date_value=existing.end_date, "
        "target_audience=existing.target_audience, max_participants=existing.max_participants, "
        "created_by_user_id=existing.created_by_user_id, created_at=existing.created_at, "
        "updated_at=existing.updated_at)\n"
        "        session.close()\n"
        "        return result\n"
        + create_program_insert
    )
    if create_program_insert in code:
        code = code.replace(create_program_insert, create_program_idempotent, 1)

    # Vaccine definitions are keyed by name.  Tool-call retries currently
    # turn a successful first create into repeated UNIQUE(name) violations.
    vaccine_insert = (
        "    db_obj = VaccineDefinition(\n"
        "        name=vd.name,\n"
        "        species=vd.species,\n"
        "        description=vd.description,\n"
        "        created_at=datetime.utcnow(),\n"
        "        updated_at=None,\n"
        "    )\n"
        "    session.add(db_obj)"
    )
    vaccine_idempotent = (
        "    existing = session.query(VaccineDefinition).filter(VaccineDefinition.name == vd.name).first()\n"
        "    if existing is not None:\n"
        "        if existing.species != vd.species or (vd.description is not None and existing.description != vd.description):\n"
        "            session.close()\n"
        "            raise HTTPException(status_code=409, detail='A vaccine definition with this name already exists with different fields')\n"
        "        resp = VaccineDefinitionResponse.from_orm(existing)\n"
        "        session.close()\n"
        "        return resp\n"
        + vaccine_insert
    )
    if vaccine_insert in code:
        code = code.replace(vaccine_insert, vaccine_idempotent, 1)

    new_code = [
        'import warnings',
        'warnings.filterwarnings("ignore", category=DeprecationWarning)',
        'from fastapi import HTTPException',
    ]
    for line in code.split("\n"):

        if 'create_engine(' in line:
            left = line.split('create_engine(')[0]
            sql_path = f"'sqlite:///{db_path}'"
            right = f"create_engine({sql_path}, connect_args={{'check_same_thread': False}})"
            line = f"{left}{right}"

        if 'uvicorn.run(app' in line:
            raw_code = f"""
            import os
            host = os.environ.get('HOST', '{config.host}')
            port = os.environ.get('PORT', {config.port})
            print(f'Server starting on port={{port}}')
            """
            lines = format_raw_code_to_lines(raw_code, indent=4)
            raw_code = f"""
            from fastapi_mcp import FastApiMCP
            mcp = FastApiMCP(app)
            mcp.mount_http()
            print("MCP server enabled, please visit http://{config.host}:{config.port}/mcp for the MCP service")
            """
            lines += format_raw_code_to_lines(raw_code, indent=4)

            line = '    uvicorn.run(app, host=host, port=int(port))'
            new_code.extend(lines)

        new_code.append(line)

    server_code = "\n".join(new_code)

    if config.temp_server_path is None:
        config.temp_server_path = os.path.join(
            os.path.dirname(config.envs_load_path),
            f"temp_server_{config.scenario.lower()}.py",
        )

    with open(config.temp_server_path, "w") as f:
        f.write(server_code)

    return config.temp_server_path


def run_server(config: Config):
    output_dir = _prepare_server_dir(config)
    db_path = _prepare_database(config, output_dir)
    server_code_path = _generate_server_code(config, db_path)

    # Save a copy of server code to output dir
    shutil.copy2(server_code_path, os.path.join(output_dir, "server_code.py"))

    logger.info(f"Server artifacts saved to {output_dir}")
    logger.info(f"Starting server for {config.scenario} on {config.host}:{config.port}")

    os.environ['PORT'] = str(config.port)
    os.environ['DATABASE_PATH'] = f"sqlite:///{db_path}"

    # Run server and capture log
    log_path = os.path.join(output_dir, "server.log")
    logger.info(f"Server log will be saved to {log_path}")
    ret = os.system(f"{sys.executable} {server_code_path} 2>&1 | tee {log_path}")

    # After server exits, save final DB snapshot
    final_db_path = os.path.join(output_dir, "final.db")
    if os.path.exists(db_path):
        shutil.copy2(db_path, final_db_path)
        logger.info(f"Saved final database to {final_db_path}")

    return ret


def start_server_process(
    scenario: str,
    envs_load_path: str,
    db_path: str,
    port: int,
    output_dir: str | None = None,
    temp_server_path: str | None = None,
) -> subprocess.Popen:
    """Start MCP server as a background subprocess.

    Returns the Popen handle so the caller can manage the process lifecycle.
    """
    cmd = [
        sys.executable, "-m", "awm.core.server",
        "--scenario", scenario,
        "--envs_load_path", envs_load_path,
        "--db_path", db_path,
        "--port", str(port),
    ]
    if output_dir:
        cmd.extend(["--output_dir", output_dir])
    if temp_server_path:
        cmd.extend(["--temp_server_path", temp_server_path])
    logger.info(f"Starting MCP server: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def run(config: Config):
    run_server(config)


if __name__ == "__main__":
    from simpleArgParser import parse_args
    config: Config = parse_args(Config)
    run(config)
