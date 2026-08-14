#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
from pathlib import Path
import re

root = Path.cwd()
chapters = sorted(root.glob("book/*/chapter-*.md"))
numbers = [int(re.search(r"chapter-(\d{2})-", path.name).group(1)) for path in chapters]
assert numbers == list(range(34)), f"chapter files are not exactly 0–33: {numbers}"
assert len(list(root.glob("book/*/README.md"))) == 7, "book must contain seven Part READMEs"

contents = (root / "CONTENTS.md").read_text(encoding="utf-8")
entries = [int(value) for value in re.findall(r"^(\d+)\. ", contents, re.MULTILINE)]
assert entries == list(range(34)), f"contents are not exactly 0–33: {entries}"

broken = []
for source in root.rglob("*.md"):
    for target in re.findall(r"\[[^]]*\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
        if "://" in target or target.startswith(("#", "mailto:")):
            continue
        destination = (source.parent / target.split("#", 1)[0]).resolve()
        if not destination.exists():
            broken.append(f"{source.relative_to(root)} -> {target}")
assert not broken, "broken Markdown links:\n" + "\n".join(broken)
print("Book structure: 7 Parts, 34 Chapters (0–33); local Markdown links resolve")
PY

python scripts/audit_ml_data.py
python -m compileall -q src examples scripts tests
pytest -q

if ! command -v composer >/dev/null; then
    echo "Composer is required for PHP validation." >&2
    exit 1
fi
composer --working-dir=php test
composer --working-dir=php lint

python examples/chapter_33_operating_harbor.py
