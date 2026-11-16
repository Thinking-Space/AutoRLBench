"""Download declared wheels during prepare and verify their exact bytes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def prepare(destination):
    specs = json.loads((Path(__file__).parent / "dependencies.json").read_text())["wheels"]
    wheels = destination / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in specs:
        path = wheels / spec["filename"]
        if not path.exists():
            subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps",
                            "--only-binary=:all:", "--dest", str(wheels),
                            spec["package"] + "==" + spec["version"]], check=True)
        if path.stat().st_size != spec["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError("Wheel identity mismatch: " + spec["filename"])
        paths.append(str(path))
    target = destination / "python"
    if not (target / ".complete").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile",
                        "--target", str(target), *paths], check=True)
        (target / ".complete").write_text(json.dumps(specs, sort_keys=True) + "\n")
    return target


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
