"""Evaluate a bounded path-query subset of Python; never execute submitted code."""
import ast
import json
from pathlib import Path
import sys

MAX_SOURCE = 65536
MAX_TEXT = 524288
MAX_ITEMS = 512


class QueryError(ValueError):
    pass


class DataPath:
    def __init__(self, root, value):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        self.root, self.path = root, path.resolve()
        if not self.path.is_relative_to(root):
            raise QueryError('Path leaves the dataset')

    def child(self, value):
        if not isinstance(value, str):
            raise QueryError('Path component must be text')
        return DataPath(self.root, self.path / value)


class Returned(Exception):
    def __init__(self, value):
        self.value = value


class Query:
    """All exposed values are JSON data, text, or a dataset-scoped path handle."""
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.fuel = 12000
        self.read_bytes = 0
        self.created_bytes = 0
        self.decoded_items = 0

    def tick(self):
        self.fuel -= 1
        if self.fuel < 0:
            raise QueryError('Query operation limit exceeded')

    def text(self, value):
        self.created_bytes += len(value.encode('utf-8'))
        if self.created_bytes > 4 * MAX_TEXT:
            raise QueryError('Text allocation budget exceeded')
        return value

    def decoded(self, value, depth=0):
        if depth > 24:raise QueryError('JSON nesting limit exceeded')
        if isinstance(value, (list, dict)):
            if len(value) > MAX_ITEMS:raise QueryError('JSON collection limit exceeded')
            self.decoded_items += len(value)
            if self.decoded_items > 8192:raise QueryError('JSON item budget exceeded')
            for child in (value.values() if isinstance(value, dict) else value):
                self.decoded(child, depth + 1)
        return value

    def path(self, value):
        if isinstance(value, DataPath):
            return value
        if not isinstance(value, str):
            raise QueryError('Path expects text')
        return DataPath(self.root, value)

    def read(self, path):
        path = self.path(path).path
        if path.stat().st_size > MAX_TEXT:
            raise QueryError('Dataset metadata file is too large')
        with path.open('rb') as stream:
            raw = stream.read(MAX_TEXT + 1)
        self.read_bytes += len(raw)
        if len(raw) > MAX_TEXT or self.read_bytes > 4 * MAX_TEXT:
            raise QueryError('Metadata read budget exceeded')
        return raw.decode('utf-8')

    def collection(self, value):
        if not isinstance(value, (list, tuple, dict)) or len(value) > MAX_ITEMS:
            raise QueryError('Expected a bounded collection')
        return value

    def bind(self, target, value, env):
        if isinstance(target, ast.Name):
            if target.id.startswith('_') or target.id in {'json', 'Path', 'len', 'str', 'FileNotFoundError', 'ValueError'}:
                raise QueryError('Reserved name')
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            self.collection(value)
            if len(target.elts) != len(value):
                raise QueryError('Unpacking length differs')
            for name, item in zip(target.elts, value):
                self.bind(name, item, env)
        else:
            raise QueryError('Only local variable assignments are supported')

    def call(self, node, env):
        args = [self.expr(x, env) for x in node.args]
        if node.keywords:
            raise QueryError('Keyword calls are not supported')
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name == 'Path' and len(args) == 1:
                return self.path(args[0])
            if name == 'len' and len(args) == 1:
                value = args[0]
                if not isinstance(value, (str, list, tuple, dict)):
                    raise QueryError('len expects data')
                return len(value)
            if name == 'str' and len(args) == 1:
                value = args[0]
                if isinstance(value, DataPath):
                    return str(value.path)
                if not isinstance(value, (str, int, bool, type(None))):
                    raise QueryError('str expects scalar data')
                return str(value)
            if name in {'FileNotFoundError', 'ValueError'} and len(args) <= 1:
                message = args[0] if args else 'Requested record unavailable'
                if not isinstance(message, str):
                    raise QueryError('Exception message must be text')
                return ('query_exception', name, message[:2000])
            raise QueryError('Unsupported function call')
        if not isinstance(node.func, ast.Attribute):
            raise QueryError('Unsupported call target')
        method = node.func.attr
        if isinstance(node.func.value, ast.Name) and node.func.value.id == 'json' and method == 'loads' and len(args) == 1:
            if not isinstance(args[0], str) or len(args[0]) > MAX_TEXT:
                raise QueryError('json.loads requires bounded text')
            self.text(args[0])  # Bound repeated JSON decoding as well as string construction.
            return self.decoded(json.loads(args[0]))
        value = self.expr(node.func.value, env)
        if isinstance(value, DataPath):
            if method in {'is_file', 'exists'} and not args:
                return value.path.is_file() if method == 'is_file' else value.path.exists()
            if method == 'read_text' and not args:
                return self.read(value)
            if method == 'joinpath' and args and all(isinstance(x, str) for x in args):
                for part in args:
                    value = value.child(part)
                return value
            if method == 'relative_to' and len(args) == 1:
                return value.path.relative_to(self.path(args[0]).path).as_posix()
        if isinstance(value, dict):
            if method == 'get' and 1 <= len(args) <= 2:
                return value.get(*args)
            if method in {'items', 'values', 'keys'} and not args:
                self.collection(value)
                return list({'items': value.items, 'values': value.values, 'keys': value.keys}[method]())
        if isinstance(value, str):
            if method in {'startswith', 'endswith'} and len(args) == 1 and isinstance(args[0], str):
                return value.startswith(args[0]) if method == 'startswith' else value.endswith(args[0])
            if method == 'replace' and len(args) == 2 and all(isinstance(x, str) for x in args):
                if not args[0] or len(value) + value.count(args[0]) * len(args[1]) > MAX_TEXT:
                    raise QueryError('Text result too large')
                return self.text(value.replace(*args))
        raise QueryError('Unsupported method: ' + method)

    def expr(self, node, env):
        self.tick()
        if isinstance(node, ast.Constant) and type(node.value) in (str, int, bool, type(None)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise QueryError('Unknown name: ' + node.id)
            return env[node.id]
        if isinstance(node, ast.Call):
            return self.call(node, env)
        if isinstance(node, ast.Subscript):
            value, key = self.expr(node.value, env), self.expr(node.slice, env)
            if not isinstance(value, (dict, list, tuple, str)):
                raise QueryError('Indexing requires data')
            return value[key]
        if isinstance(node, ast.Attribute):
            value = self.expr(node.value, env)
            if isinstance(value, DataPath) and node.attr in {'name', 'suffix'}:
                return value.path.name if node.attr == 'name' else value.path.suffix
            raise QueryError('Attribute access is not supported')
        if isinstance(node, ast.BinOp):
            left, right = self.expr(node.left, env), self.expr(node.right, env)
            if isinstance(node.op, ast.Div) and isinstance(left, DataPath):
                return left.child(right)
            if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str) and len(left) + len(right) <= MAX_TEXT:
                return self.text(left + right)
            raise QueryError('Unsupported operator')
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not self.expr(node.operand, env)
        if isinstance(node, ast.BoolOp):
            for child in node.values:
                value = self.expr(child, env)
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    return value
            return value
        if isinstance(node, ast.Compare):
            left = self.expr(node.left, env)
            for op, child in zip(node.ops, node.comparators):
                right = self.expr(child, env)
                if isinstance(op, ast.Eq): result = left == right
                elif isinstance(op, ast.NotEq): result = left != right
                elif isinstance(op, ast.In): result = left in self.collection(right)
                elif isinstance(op, ast.NotIn): result = left not in self.collection(right)
                elif isinstance(op, ast.Is): result = left is right
                elif isinstance(op, ast.IsNot): result = left is not right
                else: raise QueryError('Unsupported comparison')
                if not result: return False
                left = right
            return True
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.expr(x, env) for x in node.elts]
        if isinstance(node, ast.IfExp):
            return self.expr(node.body if self.expr(node.test, env) else node.orelse, env)
        if isinstance(node, ast.ListComp) and len(node.generators) == 1:
            generator = node.generators[0]
            if generator.is_async: raise QueryError('Async queries are unsupported')
            result = []
            for item in self.collection(self.expr(generator.iter, env)):
                local = dict(env); self.bind(generator.target, item, local)
                if all(self.expr(test, local) for test in generator.ifs):
                    result.append(self.expr(node.elt, local))
            return result
        raise QueryError('Unsupported expression: ' + type(node).__name__)

    def statements(self, nodes, env):
        for node in nodes:
            self.tick()
            if isinstance(node, ast.Return):
                raise Returned(self.expr(node.value, env))
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                self.bind(node.targets[0], self.expr(node.value, env), env)
            elif isinstance(node, ast.If):
                self.statements(node.body if self.expr(node.test, env) else node.orelse, env)
            elif isinstance(node, ast.For) and not node.orelse:
                for item in self.collection(self.expr(node.iter, env)):
                    self.bind(node.target, item, env); self.statements(node.body, env)
            elif isinstance(node, ast.Raise):
                value = self.expr(node.exc, env)
                if not isinstance(value, tuple) or value[0] != 'query_exception':
                    raise QueryError('Unsupported exception')
                if value[1] == 'FileNotFoundError': raise FileNotFoundError(value[2])
                raise QueryError(value[2])
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            else:
                raise QueryError('Unsupported statement: ' + type(node).__name__)

    def run(self, source, record):
        if len(source.encode()) > MAX_SOURCE:
            raise QueryError('Resolver source too large')
        tree = ast.parse(source)
        if sum(1 for _ in ast.walk(tree)) > 2048:
            raise QueryError('Resolver syntax too large')
        functions = []
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str): continue
            if isinstance(node, ast.Import) and all(n.name == 'json' and n.asname is None for n in node.names): continue
            if isinstance(node, ast.ImportFrom) and node.module == 'pathlib' and node.level == 0 and [(n.name,n.asname) for n in node.names] == [('Path',None)]: continue
            if isinstance(node, ast.FunctionDef) and node.name == 'resolve': functions.append(node); continue
            raise QueryError('Only json, Path and resolve(root, record) are supported')
        if len(functions) != 1:
            raise QueryError('Define exactly one resolve(root, record) function')
        func = functions[0]
        if func.decorator_list or func.args.defaults or func.args.kwonlyargs or func.args.posonlyargs or func.args.vararg or func.args.kwarg or [a.arg for a in func.args.args] != ['root','record']:
            raise QueryError('Expected resolve(root, record) without decorators or defaults')
        try:
            self.statements(func.body, {'root': str(self.root), 'record': record})
        except Returned as result:
            if not isinstance(result.value, str) or len(result.value) > 4096:
                raise QueryError('Return a relative path string')
            return result.value
        raise QueryError('Resolver returned no path')


def main():
    try:
        request = json.loads(sys.stdin.buffer.read(MAX_SOURCE))
        script = Path(sys.argv[1])
        with script.open('rb') as stream:
            source = stream.read(MAX_SOURCE + 1)
        value = Query(request['root']).run(source.decode('utf-8'), request['record'])
        result = {'ok': True, 'path': value}
    except FileNotFoundError as error:
        result = {'ok': False, 'kind': 'missing', 'message': str(error)[:2000]}
    except Exception as error:
        result = {'ok': False, 'kind': 'error', 'type': type(error).__name__, 'message': str(error)[:2000]}
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
