import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest


class StageTests(unittest.TestCase):
    def test_native_load_is_wrapped_without_disabling_optimizer(self):
        path=Path(__file__).with_name('stage.py')
        self.assertTrue(path.exists(),'Restore instrumentation missing')
        spec=importlib.util.spec_from_file_location('stage',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        source='''def initialize_model_and_optimizer(args, role):
    iteration, _ = load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    return iteration
'''
        out=m.instrument_load(source)
        ast.parse(out)
        self.assertIn('restore_load_checkpoint(args, load_checkpoint,',out)
        self.assertIn('skip_load_to_model_and_opt=False',out)

    def test_restore_does_not_advance_scheduler_a_second_time(self):
        path=Path(__file__).with_name('stage.py')
        spec=importlib.util.spec_from_file_location('stage',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        source='''def initialize_model_and_optimizer(args, role):
    iteration, _ = load_checkpoint(
        model, optimizer, opt_param_scheduler)
    if opt_param_scheduler is not None:
        opt_param_scheduler.step(increment=iteration * args.global_batch_size)
    return iteration
'''
        result=m.instrument_load(source)
        self.assertNotIn('opt_param_scheduler.step(',result)


if __name__=='__main__':unittest.main()
