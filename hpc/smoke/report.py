"""Tabulate the per-partition smoke results that ``verify_partitions.sh`` collected.

    python3 hpc/smoke/report.py <results root>

Reads ``<root>/<partition>/summary.json`` (written by smoke.py inside the job) and falls back
to the job's ``job.out`` when there is none (the job never reached the smoke test: MIG guard,
missing image, queue failure). Plain python3, no third-party imports, and nothing newer than
python 3.6: the login nodes' ``/usr/bin/python3`` is that old.
Exit 0 when every partition's required tiers passed.
"""

import json
import sys
from pathlib import Path


def tail(path: Path, lines: int = 8) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "<no job.out>"


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    root = Path(argv[1])
    parts = sorted(p for p in root.iterdir() if p.is_dir())
    if not parts:
        print(f"no partition directories under {root}")
        return 2

    all_ok = True
    letters = []
    rows = []
    for part in parts:
        summary_path = part / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            tiers = {t["tier"]: t for t in summary.get("tiers", [])}
            for letter in tiers:
                if letter not in letters:
                    letters.append(letter)
            ok = bool(summary.get("ok"))
            rows.append((part.name, summary.get("node", "?"), summary.get("gpu", "?"),
                         summary.get("image_flavor", "?"), tiers, ok, None))
        else:
            ok = False
            rows.append((part.name, "?", "?", "?", {}, False, tail(part / "job.out")))
        all_ok &= ok

    width = max(len(r[0]) for r in rows)
    header = f"{'partition':{width}}  {'result':6}  " + "  ".join(f"{l:^8}" for l in letters) + "  node / gpu / flavor"
    print(header)
    print("-" * len(header))
    for name, node, gpu, flavor, tiers, ok, fallback in rows:
        cells = []
        for letter in letters:
            t = tiers.get(letter)
            cells.append(f"{(t['status'] if t else '-'):^8}")
        print(f"{name:{width}}  {('PASS' if ok else 'FAIL'):6}  " + "  ".join(cells) + f"  {node} / {gpu} / {flavor}")
        if fallback is not None:
            print(f"{'':{width}}  no summary.json; job.out ends with:")
            for line in fallback.splitlines():
                print(f"{'':{width}}    {line}")
    print()
    print("tiers: " + "; ".join(f"{l} = {TIER_LABELS.get(l, '?')}" for l in letters))
    print(f"logs under {root}/<partition>/")
    return 0 if all_ok else 1


TIER_LABELS = {
    "a": "torch + kernels for this GPU",
    "b": "Kit boots headless",
    "c": "Forge env steps",
    "d": "short training run",
    "e": "camera + mp4 (needs RT cores; informational)",
}

if __name__ == "__main__":
    sys.exit(main(sys.argv))
