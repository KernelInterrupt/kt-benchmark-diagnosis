#!/usr/bin/env python3
"""Report interpreter, optional packages, and shipped CLI usability."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "torch", "PIL", "cudf", "cupy", "pyyaml"]
def discover_clis(root: Path) -> list[str]:
    files = []
    for folder in (root / "examples", root / "scripts", root / "tools"):
        for path in sorted(folder.glob("*.py")):
            if path.name == "env_check.py":
                continue
            if "argparse" in path.read_text(encoding="utf-8", errors="ignore"):
                files.append(str(path.relative_to(root)))
    return files


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    print(f"interpreter: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    for name in PACKAGES:
        module = {"scikit-learn": "sklearn", "pyyaml": "yaml"}.get(name, name)
        print(f"package {name}: {'importable' if importlib.util.find_spec(module) else 'missing'}")
    for rel in discover_clis(root):
        path = root / rel
        try:
            proc = subprocess.run([sys.executable, str(path), "--help"], cwd=root, text=True, capture_output=True, timeout=30)
            if proc.returncode == 0:
                print(f"cli {rel}: runnable")
            else:
                detail = (proc.stderr or proc.stdout).strip().splitlines()[-1] if (proc.stderr or proc.stdout).strip() else "unknown error"
                print(f"cli {rel}: not runnable; {detail}")
        except Exception as exc:
            print(f"cli {rel}: not runnable; {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
