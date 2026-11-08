"""Native TaskRunner with observation-only hooks installed in its Ray process."""
import ray
from verl.trainer.main_ppo_v0 import BaseTaskRunner, TaskRunner
from observer import install_trainer_hooks

NATIVE_RUN = TaskRunner.__ray_metadata__.modified_class.run


@ray.remote(num_cpus=1)
class ObservedTaskRunner(BaseTaskRunner):
    def run(self, config):
        install_trainer_hooks()
        return NATIVE_RUN(self, config)
