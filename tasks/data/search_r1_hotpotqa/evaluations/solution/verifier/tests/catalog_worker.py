"""Execute only the supported pure producer function in a bounded subprocess."""
import argparse
import ast
import builtins
import json
from pathlib import Path
import re
import resource
import sys

NODES = {
    'FunctionDef', 'arguments', 'arg', 'Return', 'Assign', 'AugAssign', 'Name',
    'Load', 'Store', 'For', 'If', 'IfExp', 'Expr', 'Call', 'Attribute',
    'Constant', 'List', 'Tuple', 'Dict', 'Set', 'Subscript', 'Slice',
    'BinOp', 'BoolOp', 'UnaryOp', 'Compare', 'ListComp', 'DictComp', 'SetComp',
    'GeneratorExp', 'comprehension', 'JoinedStr', 'FormattedValue', 'keyword',
    'Add', 'Mult', 'Sub', 'Div', 'FloorDiv', 'Mod', 'And', 'Or', 'Not',
    'USub', 'UAdd', 'Eq', 'NotEq', 'In', 'NotIn', 'Is', 'IsNot',
    'Lt', 'LtE', 'Gt', 'GtE', 'Pass', 'Break', 'Continue', 'Raise',
}
METHODS = {'get', 'strip', 'append', 'extend', 'copy', 'items', 'keys', 'values',
           'split', 'join', 'startswith', 'endswith', 'replace', 'findall'}
BUILTINS = ('list', 'dict', 'set', 'tuple', 'enumerate', 'len', 'str', 'range',
            'min', 'max', 'zip', 'all', 'any', 'sorted', 'sum', 'isinstance',
            'ValueError', 'TypeError')


def extract(source):
    if len(source.encode()) > 65536:
        raise ValueError('Producer source exceeds 64 KiB')
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build']
    if len(functions) != 1:
        raise ValueError('Exactly one build function is required')
    function = functions[0]
    args = function.args
    if (function.decorator_list or function.returns or args.defaults or args.kw_defaults
            or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
            or [a.arg for a in args.args] != ['rows', 'passages']
            or any(a.annotation for a in args.args)):
        raise ValueError('Expected a plain build(rows, passages) function')
    nodes = list(ast.walk(function))
    if len(nodes) > 2000:
        raise ValueError('Producer AST exceeds its budget')
    for node in nodes:
        if type(node).__name__ not in NODES:
            raise ValueError('Unsupported producer syntax: ' + type(node).__name__)
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError('Private names are unsupported')
        if (isinstance(node, ast.Attribute) and node.attr not in METHODS
                and not (isinstance(node.value, ast.Name) and node.value.id == 're' and node.attr == 'S')):
            raise ValueError('Unsupported producer method: ' + node.attr)
        if isinstance(node, ast.Call) and not (
                isinstance(node.func, ast.Name) and node.func.id in BUILTINS
                or isinstance(node.func, ast.Attribute) and node.func.attr in METHODS):
            raise ValueError('Unsupported producer call')
        if isinstance(node, ast.comprehension) and node.is_async:
            raise ValueError('Async producers are unsupported')
    return tree, function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
    _, function = extract(args.source.read_text())
    scope = {'__builtins__': {n: getattr(builtins, n) for n in BUILTINS}, 're': re}
    program = ast.Module(body=[function], type_ignores=[])
    exec(compile(program, '<submitted-producer>', 'exec'), scope)
    inputs = json.loads(args.inputs.read_text())
    budget = [2_000_000]

    def trace(frame, event, value):
        if frame.f_code.co_filename == '<submitted-producer>' and event == 'line':
            budget[0] -= 1
            if budget[0] < 0:
                raise ValueError('Producer operation budget exceeded')
        return trace

    sys.settrace(trace)
    try:
        output = scope['build'](inputs['rows'], inputs['passages'])
    finally:
        sys.settrace(None)
    with args.output.open('x') as stream:
        json.dump(output, stream, ensure_ascii=False, allow_nan=False)


if __name__ == '__main__':
    main()
