import os
from dataclasses import dataclass
from loguru import logger
from awm.tools import tools_jsonl_load, normalize_scenario_name
from awm.core.env import batch_test_environments


@dataclass
class Config:
    # path to gen_envs.jsonl to test
    input: str
    # filter specific scenarios to test
    allowed_scenarios: list[str] | None = None

    def pre_process(self):
        assert os.path.exists(self.input), f"Environment file {self.input} not found"
        if self.allowed_scenarios:
            self.allowed_scenarios = {normalize_scenario_name(name) for name in self.allowed_scenarios}


def run(config: Config):
    env_configs = tools_jsonl_load(config.input)

    if config.allowed_scenarios is not None:
        env_configs = [e for e in env_configs if normalize_scenario_name(e["scenario"]) in config.allowed_scenarios]

    if not env_configs:
        logger.warning("No environments to test.")
        return

    logger.info(f"Test mode: loaded {len(env_configs)} environments from {config.input}")

    test_results = batch_test_environments(env_configs, max_workers=min(os.cpu_count() or 4, len(env_configs), 64))

    passed = []
    failed = []
    for env_config, (success, output, _) in zip(env_configs, test_results):
        scenario_name = env_config["scenario"]
        if success:
            passed.append(scenario_name)
        else:
            failed.append((scenario_name, output))

    for name in passed:
        logger.success(f"PASSED: {name}")

    logger.info(f"\n{'='*100}\n\n")

    for name, output in failed:
        error_preview = (output or "")[:500]
        logger.error(f"FAILED: {name}\n    {error_preview}")

    logger.info(f"\n{'='*100}\nTest Results: {len(passed)} passed, {len(failed)} failed out of {len(env_configs)} total\n{'='*100}")
