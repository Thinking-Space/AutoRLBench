import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_cleanup import clean_source, is_observation


def test_observation_removal_keeps_arithmetic_branches_and_docstring():
    text = '''# SPDX-License-Identifier: Apache-2.0
# explanatory runtime hint
def run(x):
    """Native function contract."""
    logger.info("diagnosis=%s", x)
    if x:
        print("secret", x)
    return x + 2  # diagnostic comment
'''
    out = clean_source(text)
    assert 'diagnosis' not in out and 'secret' not in out and 'runtime hint' not in out
    assert '# SPDX' in out and 'Native function contract.' in out
    namespace = {}
    exec(out, namespace)
    assert namespace['run'](3) == 5
    assert not any(is_observation(n) for n in ast.walk(ast.parse(out)))


def test_unicode_logging_keeps_statement_boundaries():
    text = 'def f(x):\n    if x:\n        print("输出 →", x)\n    else:\n        print("空")\n    return x\n'
    ns = {}
    exec(clean_source(text), ns)
    assert ns['f'](2) == 2


def test_dormant_backward_dump_cannot_register_a_hook(tmp_path, monkeypatch):
    source = '''def _maybe_dump_top_logprob_backward(name, tensor):
    if __import__('os').environ.get("MILES_LOGPROB_BACKWARD_DEBUG_DIR"):
        tensor.register_hook(lambda grad: __import__('pathlib').Path("dump.pt").write_text(str(grad)))
'''
    class Tensor:
        def register_hook(self, hook):
            raise AssertionError('A dormant diagnostic hook survived')
    ns = {}
    exec(clean_source(source), ns)
    monkeypatch.setenv('MILES_LOGPROB_BACKWARD_DEBUG_DIR', str(tmp_path))
    assert ns['_maybe_dump_top_logprob_backward']('test', Tensor()) is None
    assert not list(tmp_path.iterdir())
