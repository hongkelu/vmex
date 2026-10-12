"""Resolution ladder of the three-term free-boundary re-solve: for each set in runs/res_check_list.txt print QS error,
B.n residual, iota and aspect ratio at 8x8/VC48, 12x12/VC64, 12x12/VC64/NS99 and 16x16/VC96, with ratios to 12x12.
    python analysis/res_check_report.py"""
import json
from pathlib import Path
P = Path(__file__).resolve().parents[1]
RES = [("8x8", "8x8/VC48"), ("12x12", "12x12/VC64"), ("12x12ns", "12x12/VC64/NS99"), ("16x16", "16x16/VC96")]
names = [l.strip() for l in (P / "runs/res_check_list.txt").read_text().splitlines() if l.strip()]
print(f"{'set':20s} " + " ".join(f"{lab:>18s}" for _, lab in RES))
for n in names:
    rows = {}
    for key, _ in RES:
        f = P / f"runs/lpqa_vacuum_eval_{key}" / n / "row.json"
        if f.exists():
            rows[key] = json.load(open(f))
    ref = rows.get("12x12")
    def cell(key, field, fmt):
        r = rows.get(key)
        if r is None: return f"{'-':>18s}"
        v = r[field]; s = fmt % v
        if ref is not None and key != "12x12" and ref[field]: s += f" ({v / ref[field]:.2f})"
        return f"{s:>18s}"
    print(f"{n:20s} " + " ".join(cell(k, "qa_total", "%.3e") for k, _ in RES) + "   qa")
    print(f"{'':20s} " + " ".join(cell(k, "bn", "%.2e") for k, _ in RES) + "   bn")
    print(f"{'':20s} " + " ".join(cell(k, "iota_edge", "%.4f") for k, _ in RES) + "   iota_edge")
    print(f"{'':20s} " + " ".join(cell(k, "seconds", "%.0f") for k, _ in RES) + "   s")
