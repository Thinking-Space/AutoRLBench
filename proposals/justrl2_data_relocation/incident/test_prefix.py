"""Prefix-stop contract. Fixtures exercise control flow, not model training."""
import ast
import importlib.util
from pathlib import Path
import unittest


class PrefixTests(unittest.TestCase):
    def test_stop_is_after_completed_iteration_and_before_next_generation(self):
        path = Path(__file__).with_name('prefix.py')
        self.assertTrue(path.exists(), 'Prefix preservation helper missing')
        spec = importlib.util.spec_from_file_location('prefix', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = '''async def train(args):
    for rollout_id in range(8):
        await generate(rollout_id)
        await train_step(rollout_id)
        if rollout_id == 3:
            await save(rollout_id)
    await rollout_manager.dispose.remote()
'''
        result = module.stop_after_checkpoint(source, 3)
        tree = ast.parse(result)
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For))
        self.assertIsInstance(loop.body[-1], ast.If)
        self.assertEqual(ast.unparse(loop.body[-1].test), 'rollout_id == 3')
        self.assertIsInstance(loop.body[-1].body[0], ast.Break)
        self.assertIn('await save(rollout_id)', result)


if __name__ == '__main__':
    unittest.main()
