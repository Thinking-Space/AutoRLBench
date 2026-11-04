import os
from dataclasses import dataclass
from loguru import logger

from awm.core.scenario import Config as ScenarioConfig, run as run_scenario
from awm.core.task import Config as TaskConfig, run as run_task
from awm.core.db import Config as DbConfig, run as run_db
from awm.core.sample import Config as SampleConfig, run as run_sample
from awm.core.spec import Config as SpecConfig, run as run_spec
from awm.core.env import Config as EnvConfig, run as run_env
from awm.core.verifier import Config as VerifierConfig, VerificationMode, run as run_verifier


@dataclass
class Config:
    # seed scenario file path
    input: str = "./outputs/seed_scenario.jsonl"
    # base directory for all outputs
    output_dir: str = "outputs"
    # number of scenarios to generate
    target_count: int = 1000
    # tasks per scenario
    num_tasks: int = 10
    # LLM model to use
    model: str = "your-llm-model-name"
    # verification mode: sql or code
    verifier_mode: VerificationMode = VerificationMode.sql

    def pre_process(self):
        assert os.path.exists(self.input), f"Seed scenario file {self.input} not found"
        os.makedirs(self.output_dir, exist_ok=True)

        if os.environ.get('AWM_SYN_OVERRIDE_MODEL'):
            self.model = os.environ.get('AWM_SYN_OVERRIDE_MODEL')
        
        assert self.model != "your-llm-model-name", "Please set the model name in the environment variable AWM_SYN_OVERRIDE_MODEL"


def run(config: Config):
    d = config.output_dir

    scenario_output = f"{d}/gen_scenario.jsonl"
    task_output = f"{d}/gen_tasks.jsonl"
    db_output = f"{d}/gen_db.jsonl"
    sample_output = f"{d}/gen_sample.jsonl"
    spec_output = f"{d}/gen_spec.jsonl"
    env_output = f"{d}/gen_envs.jsonl"
    verifier_output = f"{d}/gen_verifier.jsonl"
    database_dir = f"{d}/databases"

    # Step 1: Scenario Generation
    logger.info("\n" + "=" * 80)
    logger.info("Step 1/7: Generating scenarios...")
    logger.info("=" * 80)
    scenario_config = ScenarioConfig(
        input_path=config.input,
        output_path=scenario_output,
        target_count=config.target_count,
        model=config.model,
    )
    scenario_config.pre_process()
    run_scenario(scenario_config)

    # Step 2: Task Generation
    logger.info("\n" + "=" * 80)
    logger.info("Step 2/7: Generating tasks...")
    logger.info("=" * 80)
    task_config = TaskConfig(
        input=scenario_output,
        output=task_output,
        num_tasks=config.num_tasks,
        model=config.model,
    )
    task_config.pre_process()
    run_task(task_config)

    # Step 3: Database Schema Generation
    logger.info("\n" + "=" * 80)
    logger.info("Step 3/7: Generating database schemas...")
    logger.info("=" * 80)
    db_config = DbConfig(
        input=task_output,
        output=db_output,
        database_dir=database_dir,
        model=config.model,
    )
    db_config.pre_process()
    run_db(db_config)

    # Step 4: Sample Data Generation
    logger.info("\n" + "=" * 80)
    logger.info("Step 4/7: Generating sample data...")
    logger.info("=" * 80)
    sample_config = SampleConfig(
        input_task=task_output,
        input_db=db_output,
        output=sample_output,
        database_dir=database_dir,
        model=config.model,
    )
    sample_config.pre_process()
    run_sample(sample_config)

    # Step 5: API Spec Generation
    logger.info("\n" + "=" * 80)
    logger.info("Step 5/7: Generating API specifications...")
    logger.info("=" * 80)
    spec_config = SpecConfig(
        input_task=task_output,
        input_db=db_output,
        output=spec_output,
        model=config.model,
    )
    spec_config.pre_process()
    run_spec(spec_config)

    # Step 6: Environment Code Generation
    logger.info("\n" + "=" * 80)
    logger.info("Step 6/7: Generating environment code...")
    logger.info("=" * 80)
    env_config = EnvConfig(
        input_spec=spec_output,
        input_db=db_output,
        output=env_output,
        database_dir=database_dir,
        model=config.model,
    )
    env_config.pre_process()
    run_env(env_config)

    # Step 7: Verification Code Generation
    logger.info("\n" + "=" * 80)
    logger.info("Step 7/7: Generating verification code...")
    logger.info("=" * 80)
    verifier_config = VerifierConfig(
        input_task=task_output,
        output=verifier_output,
        database_dir=database_dir,
        mode=config.verifier_mode,
        model=config.model,
    )
    verifier_config.pre_process()
    run_verifier(verifier_config)

    logger.success("\n" + "=" * 80)
    logger.success("Full pipeline complete!")
    logger.success(f"Output directory: {d}")
    logger.success(f"  Scenarios:     {scenario_output}")
    logger.success(f"  Tasks:         {task_output}")
    logger.success(f"  DB Schemas:    {db_output}")
    logger.success(f"  Sample Data:   {sample_output}")
    logger.success(f"  API Specs:     {spec_output}")
    logger.success(f"  Environments:  {env_output}")
    logger.success(f"  Verifiers:     {verifier_output}")
    logger.success(f"  Databases:     {database_dir}/")
    logger.success("=" * 80)
