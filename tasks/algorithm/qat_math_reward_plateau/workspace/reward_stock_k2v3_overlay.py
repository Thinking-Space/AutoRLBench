"""Stage the historical K2V3 compatibility changes in scratch; codex."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(overlay, target):
    original = Path(importlib.util.find_spec("vllm").origin).parent
    pins = json.loads((overlay / "manifest.json").read_bytes())
    changed = {p.relative_to(overlay) for p in overlay.rglob("*.py")
               if p.relative_to(overlay).as_posix() in pins
               and sha(p) != pins[p.relative_to(overlay).as_posix()]}
    if len(changed) != 5:
        raise ValueError("Expected exactly four registrations and one CUDA pack update")
    for rel, digest in pins.items():
        if sha(original / rel) != digest:
            raise ValueError(f"Stock source hash drift: {rel}")

    def mirror(src, dest, relative):
        dest.mkdir()
        for item in src.iterdir():
            rel = relative / item.name
            out = dest / item.name
            if rel in changed:
                raw = (overlay / rel).read_bytes()
                compile(raw, str(rel), "exec")
                out.write_bytes(raw)
            elif any(rel in path.parents for path in changed):
                mirror(item, out, rel)
            else:
                out.symlink_to(item, target_is_directory=item.is_dir())

    target.mkdir(parents=True, exist_ok=False)
    mirror(original, target / "vllm", Path())
    for rel in changed:
        print(rel, sha(target / "vllm" / rel), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    stage(args.overlay.resolve(), args.target.resolve())
    if args.command:
        command = args.command[1:] if args.command[0] == "--" else args.command
        if not command:
            raise ValueError("Empty command")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(args.target.resolve()), env.get("PYTHONPATH"))))
        os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
