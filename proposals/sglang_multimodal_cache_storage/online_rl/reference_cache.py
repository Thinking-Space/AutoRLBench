"""Apply the reference change inside the source-pinned cache insertion method."""

from __future__ import annotations

import ast
from dataclasses import replace


def reference_class(cache_class, source_path):
    original = source_path.read_text()
    insertion = "        self.mm_cache[mm_hash] = embedding"
    if original.count(insertion) != 1:
        raise ValueError("reference change does not apply to this cache revision")
    fixed = original.replace(insertion, """        tensor = embedding.embedding
        if tensor.untyped_storage().nbytes() > data_size:
            embedding = replace(embedding, embedding=tensor.clone(memory_format=torch.contiguous_format))
""" + insertion)
    cache = next(node for node in ast.parse(fixed).body
                 if isinstance(node, ast.ClassDef) and node.name == "MultiModalStaticCache")
    method = next(node for node in cache.body if isinstance(node, ast.FunctionDef) and node.name == "set")
    namespace = {**cache_class.set.__globals__, "replace": replace}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
    return type("ReferenceCache", (cache_class,), {"set": namespace["set"]})
