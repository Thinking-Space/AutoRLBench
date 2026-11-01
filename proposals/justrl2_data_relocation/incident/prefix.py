"""Construct the real step-3 prefix while keeping the eight-step schedule."""
import ast


def stop_after_checkpoint(source, iteration=3):
    tree = ast.parse(source)
    candidates = [n for n in ast.walk(tree) if isinstance(n, ast.For)
                  and isinstance(n.target, ast.Name) and n.target.id == 'rollout_id']
    if len(candidates) != 1:
        raise ValueError('Expected one native rollout loop')
    loop = candidates[0]
    lines = source.splitlines(keepends=True)
    lines.insert(loop.end_lineno, ' ' * (loop.col_offset + 4) + f'if rollout_id == {iteration}:\n'
                 + ' ' * (loop.col_offset + 8) + 'break\n')
    result = ''.join(lines)
    ast.parse(result)
    return result
