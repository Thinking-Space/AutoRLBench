# SPDX-License-Identifier: Apache-2.0
"""Remove supplied observations from contestant copies, preserving computation."""
import ast
import io
import tokenize
import shutil

LOG_METHODS = {'debug', 'info', 'warning', 'warn', 'error', 'exception', 'critical', 'log'}
DIAGNOSTIC_FUNCTIONS = {
    '_maybe_dump_top_logprob_backward', '_log_extreme_logdiff_diagnostic',
    'maybe_dump_policy_loss_debug', 'save_debug_train_data', 'safe_torch_save_debug',
    '_save_debug_rollout_data',
}


def strip_diagnostic_helpers(text):
    """Disable pinned diagnostic helpers, including dormant backward file hooks.

    Preserve signatures for native imports but remove their implementation. The
    named functions only emit observations; none returns a training value.
    """
    tree = ast.parse(text)
    selected = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in DIAGNOSTIC_FUNCTIONS]
    lines = text.splitlines(keepends=True)
    for node in sorted(selected, key=lambda n: n.lineno, reverse=True):
        first, last = node.body[0], node.body[-1]
        # Pinned helper bodies are ordinary indented suites.
        lines[first.lineno - 1:last.end_lineno] = [' ' * first.col_offset + 'return None\n']
    return ''.join(lines)


def remove_diagnostic_tools(workspace):
    """Remove standalone dump browsers and retain inert import-compatible APIs."""
    directory = workspace / 'miles/utils/debug_utils'
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    (directory / '__init__.py').write_text('')
    (directory / 'icepop_dump.py').write_text(
        'def record_icepop_scatter(*args, **kwargs):\n    return None\n'
        'def flush_icepop_scatter(*args, **kwargs):\n    return None\n'
        'def set_current_train_step(*args, **kwargs):\n    return None\n')
    (workspace / 'miles/utils/train_dump_utils.py').write_text(
        'def safe_torch_save_debug(*args, **kwargs):\n    return None\n'
        'def save_debug_train_data(*args, **kwargs):\n    return None\n')
    (workspace / 'miles/backends/training_utils/debug_dump.py').write_text(
        'def maybe_dump_policy_loss_debug(*args, **kwargs):\n    return None\n')
    (workspace / 'miles/utils/dumper_utils.py').write_text(
        'import enum\nclass DumperPhase(enum.Enum):\n'
        '    INFERENCE = "inference"\n    FWD_ONLY = "fwd_only"\n    FWD_BWD = "fwd_bwd"\n'
        'def get_sglang_env(*args, **kwargs):\n    return {}\n'
        'async def configure_sglang(*args, **kwargs):\n    return None\n'
        'class DumperMegatronUtil:\n'
        '    def __init__(self, *args, **kwargs):\n        pass\n'
        '    def wrap_forward_step(self, function):\n        return function\n'
        '    def finalize(self, *args, **kwargs):\n        return None\n')


def is_observation(node):
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    call = node.value.func
    if isinstance(call, ast.Name) and call.id == 'print':
        return True
    if not isinstance(call, ast.Attribute) or call.attr not in LOG_METHODS:
        return False
    target = call.value
    name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else ''
    return name.lower().endswith('logger') or name in ('logging', 'log')


def clean_source(text):
    """Strip standalone print/logger calls and non-license comments uniformly.

    Keep native identifiers, docstrings, branches, asserts and return values.
    Functions are never replaced with reference implementations.
    """
    text = strip_diagnostic_helpers(text)
    lines = text.splitlines(keepends=True)
    nodes = [n for n in ast.walk(ast.parse(text)) if is_observation(n)]
    for node in sorted(nodes, key=lambda n: (n.lineno, n.col_offset), reverse=True):
        # Keep an empty suite valid; no descriptive instrumentation marker.
        before = lines[node.lineno - 1].encode()[:node.col_offset].decode()
        after = lines[node.end_lineno - 1].encode()[node.end_col_offset:].decode()
        if before.strip() or after.strip():
            # Native source pins contain no inline observation side effects.
            # Avoid silently deleting adjacent executable statements.
            raise ValueError('Inline logger/print statement needs explicit review')
        lines[node.lineno - 1:node.end_lineno] = [before + 'pass' + after]
    result = ''.join(lines)
    lines = result.splitlines(keepends=True)
    comments = [t for t in tokenize.generate_tokens(io.StringIO(result).readline)
                if t.type == tokenize.COMMENT and not any(word in t.string for word in ('SPDX', 'Copyright', 'copyright'))]
    for token in reversed(comments):
        row, column = token.start
        line = lines[row - 1]
        lines[row - 1] = line[:column] + line[token.end[1]:]
    result = ''.join(lines)
    ast.parse(result)
    return result
