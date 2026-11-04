#!/usr/bin/env python3
"""Offline check: a freshly built critic value head must predict exactly zero.

A critic that has not yet seen a single reward has no information with which to
prefer any value estimate over another, so the only defensible initial state is
V(s) == 0 for every s. Anything else is an arbitrary prediction that the warmup
has to spend its first steps unlearning, and it shows up as an inflated
`train/critic-value_loss` that has nothing to do with the task.

This is the verifier's copy (layer 4 of the production fix, relocated): the
check used to ship inside the workspace as `tools/check_value_head.py`, but a
workspace-shipped checker that names the defect and explains its mechanism
tells the issue-level agent the diagnosis, so it lives here instead, where the
agent cannot read it.

Runs on CPU in a couple of seconds: torch only -- no GPU, no megatron, no
checkpoint, no distributed init.

    python3 check_value_head.py path/to/model_provider.py ...

Exit status:

    0   every checked file yields an all-zero value head
    1   at least one does not -- the condition this check exists to catch
    2   the check could not be run at all (torch missing, nothing to check,
        or the module under test would not import)

1 and 2 are kept apart on purpose. Collapsing them reports a missing
dependency as a failed check, which is the same failure mode as the one at
issue: a number presented as a measurement of something it never measured.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types

HIDDEN_SIZE = 2048  # Qwen3-1.7B; the check is shape independent.

# model_provider.py imports megatron and slime at module scope, but
# LinearForLastLayer itself touches neither: it is a torch.nn.Linear subclass
# that reads two scalars off the config. Standing in for those imports keeps
# this check runnable outside the training image -- the alternative is a
# checker you can only run where the bug has already cost you a run.
_STUB_ROOTS = ("megatron", "slime")


class CannotRun(RuntimeError):
    """The check could not be performed.

    Kept distinct from the check *failing*, because the two call for opposite
    responses and only one of them is about the value head. Callers map this to
    exit 2 ("could not run"), never to exit 1 ("the value head is non-zero").
    """


class _Stub(types.ModuleType):
    """A module whose every attribute is another stub, created on demand."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        child = _Stub(f"{self.__name__}.{name}")
        child.__path__ = []
        setattr(self, name, child)
        sys.modules[child.__name__] = child
        return child


class _StubFinder:
    """Resolve any `import megatron.*` / `import slime.*` to a stub."""

    def __init__(self, roots: tuple[str, ...]) -> None:
        self.roots = roots

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        if fullname.split(".")[0] not in self.roots:
            return None
        return importlib.util.spec_from_loader(fullname, self, is_package=True)

    def create_module(self, spec):
        module = _Stub(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):  # nothing to execute
        pass


def install_stub_finder(force: bool = False) -> None:
    """Make `import megatron.*` / `import slime.*` resolve, however little is installed.

    Appended to the end of `sys.meta_path`, so the real finders are always tried
    first and only names they cannot resolve reach the stub. That granularity is
    per *module*, not per top-level package, which is what this needs: the
    training image ships `megatron.core` (a real package, used for real) but not
    necessarily `megatron.training`, and a check that keyed off whether
    `megatron` imports would conclude everything was present and then die on the
    first missing submodule.

    With `force`, the finder goes to the front instead and shadows the real
    packages outright -- the fallback for when a real module is importable but
    blows up on exec because one of *its* dependencies is missing.
    """
    finder = _StubFinder(_STUB_ROOTS)
    if force:
        sys.meta_path.insert(0, finder)
        for name in list(sys.modules):
            if name.split(".")[0] in _STUB_ROOTS:
                del sys.modules[name]
    else:
        sys.meta_path.append(finder)


def default_targets() -> list[str]:
    # The verifier always passes an explicit path (the workspace under test);
    # with no argument there is nothing sensible to default to from here, the
    # file being verifier-side rather than workspace-side.
    return []


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("_mp_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_head(path: str):
    install_stub_finder()
    try:
        module = _load_module(path)
    except Exception:  # noqa: BLE001 -- see below; narrowed on the retry
        # A real megatron/slime module was found but could not be executed: a
        # partial install whose own imports fail (ImportError), or one whose
        # files this user cannot read (PermissionError, which is what running
        # the image as a non-root uid gives you for /root/Megatron-LM).
        # Nothing here needs the real packages -- only the class under test --
        # so shadow them outright and retry.
        install_stub_finder(force=True)
        try:
            module = _load_module(path)
        except Exception as exc:  # noqa: BLE001
            raise CannotRun(f"{path} could not be imported: {exc!r}") from exc
    # LinearForLastLayer reads only these two attributes off the config.
    config = types.SimpleNamespace(
        sequence_parallel=False,
        init_method_std=0.02,
        hidden_size=HIDDEN_SIZE,
    )
    return module.LinearForLastLayer(HIDDEN_SIZE, 1, config=config)


def check(path: str) -> bool:
    head = build_head(path)
    weight = head.weight.data
    bias = None if head.bias is None else head.bias.data

    weight_max = float(weight.abs().max())
    bias_max = 0.0 if bias is None else float(bias.abs().max())
    ok = weight_max == 0.0 and bias_max == 0.0

    print(f"{'PASS' if ok else 'FAIL'}  {path}")
    print(f"        value head shape      {tuple(weight.shape)}")
    print(f"        max |weight|          {weight_max:.6g}")
    print(f"        max |bias|            {bias_max:.6g}")
    if not ok:
        # What this costs at the start of warmup. The value head is a linear
        # projection of the final hidden state, so std(V) ~ |w| * ||h||, and
        # the first value loss is mean((V - R)^2) ~ E[V^2] + E[R^2].
        rms = float((weight.double() ** 2).mean().sqrt()) * HIDDEN_SIZE**0.5
        print(f"        => a fresh critic predicts V != 0 (||w||_2 = {rms:.4g}),")
        print("           so the warmup value loss starts at E[V^2] + E[R^2]")
        print("           instead of E[R^2]. Initialise the value head to zero.")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="model_provider.py files to check")
    args = ap.parse_args()

    # torch is the one import this check genuinely needs: it builds the real
    # layer and reads the weights back. Probe it up front and return 2 ("could
    # not run"), rather than letting the ImportError surface from inside
    # build_head, where exit 1 would be indistinguishable from the value head
    # really being non-zero. Both callers already map 2 to "needs torch" -- but
    # without this they never see it, and a missing dependency gets reported as
    # a failed fix. That is the same class of mistake this whole case is about:
    # a number that is not measuring what it claims to measure.
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        print(f"cannot run this check: {exc}", file=sys.stderr)
        print(
            "it constructs the real value head, so torch must be importable; "
            "run it inside the training image.",
            file=sys.stderr,
        )
        return 2

    targets = args.paths or default_targets()
    if not targets:
        print("no model_provider.py to check", file=sys.stderr)
        return 2

    try:
        results = [check(path) for path in targets]
    except CannotRun as exc:
        print(f"cannot run this check: {exc}", file=sys.stderr)
        return 2
    print()
    if all(results):
        print(f"value head zero-initialisation OK ({len(results)} file(s) checked)")
        return 0
    print(f"value head is NOT zero-initialised in {results.count(False)} of {len(results)} file(s)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
