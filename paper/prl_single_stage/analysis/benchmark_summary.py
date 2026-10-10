"""LP QA vacuum benchmark summary: published stage-two coil sets vs our free-boundary single stage.

    python analysis/benchmark_summary.py [analysis/benchmark_summary.md]

Rows: L18, L20, L24 (Gil and Wechsung starts), 3 coils.  Columns: QS of the equilibrium the published coils make
(runs/lpqa_vacuum_eval), ours under the driver's margin constraints (stricter than theirs), ours under exactly their
thresholds (the *-exact / *-feas2 twins), and the improvement factors.  Values are the last metrics.jsonl row of the most
advanced continuation; "steps" counts every continuation.
"""
import json
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]


def ext(name):
    r = json.load(open(P / "runs/lpqa_vacuum_eval" / name / "row.json"))
    return r["qa_total"]


def chain(prefix):
    """(final qa, best qa, total steps, run name) over prefix, prefix-c1, prefix-c2, ..."""
    total, final, best, name = 0, None, None, None
    for d in sorted((p for p in (P / "runs/batch").glob(f"{prefix}*") if p.is_dir()
                     and re.fullmatch(re.escape(prefix) + r"(-c\d+)?", p.name)),
                    key=lambda p: int(m.group(1)) if (m := re.search(r"-c(\d+)$", p.name)) else 0):
        rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()] if (d / "metrics.jsonl").exists() else []
        if not rows:
            continue
        total += rows[-1]["step"]
        final, name = rows[-1]["qa"], d.name
        b = min(r["qa"] for r in rows)
        best = b if best is None else min(best, b)
    return final, best, total, name


rows = [("L18, Gil start", "gil_L18", "vac-L18", "vac-L18-exact"),
        ("L18, Wechsung start", "wechsung_L18", "vac-L18-wech", None),
        ("L20, Gil start", "gil_L20", "vac-L20", "vac-L20-exact"),
        ("L24, Gil start", "gil_L24", "vac-L24", "vac-L24-feas2"),
        ("L24, Wechsung start", "wechsung_L24", "vac-L24-wech", "vac-L24-wech-feas2"),
        ("L24, Wechsung start, coil order 24", "wechsung_L24", None, "vac-L24-wech-o24-feas"),
        ("3 coils (18 m), Gil start", "gil_3coil_L18", "vac-3coil", "vac-3coil-exact")]
out = ["| row | published coils | ours, stricter (margins) | steps | ours, exactly their thresholds | steps | gain (exact) |",
       "|---|---|---|---|---|---|---|"]
for label, pub, margin, exact in rows:
    q0 = ext(pub)
    m = chain(margin) if margin else (None, None, 0, None)
    e = chain(exact) if exact else (None, None, 0, None)
    f = lambda v: "–" if v is None else f"{v:.2e}"
    gain = "–" if e[0] is None else f"{q0 / e[0]:.1f}×"
    out.append(f"| {label} | {q0:.2e} | {f(m[0])} | {m[2]} | {f(e[0])} | {e[2]} | {gain} |")
text = "\n".join(out)
print(text)
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(text + "\n")
