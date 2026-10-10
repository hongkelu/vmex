"""Table II (Supplement): the LP QA vacuum benchmark in Gil et al.'s format, published sets and ours.

    python analysis/table2_gil_format.py [analysis/table2_gil_format.md]

Columns: L per half period (m), max curvature (1/m), max MSC (1/m^2), min coil-coil (m), min coil-surface (m),
<|B.n|>/<|B|> on the LP QA target (their definition) and on the equilibrium's own LCFS, QS by their pipeline
(QFM surface -> fixed-boundary solve -> two-term residual on 51 surfaces), QS of the three-term free-boundary
equilibrium at the common 12x12 resolution (our metric), max force (MN/m per turn, their regularisation).
Reactor scale (R0 = 10.1266 m) for the engineering columns.  Sources: runs/published/<set>/gil_score_step0,
runs/batch/<run>/gil_score_step<N> (latest), runs/lpqa_vacuum_eval_12x12/{<set>, ours_<run>}/row.json,
references/gil2026/coils_npz/summary.json for the values printed in their paper.
"""
import json
import re
import sys
from pathlib import Path

P = Path(__file__).resolve().parents[1]
ROWS = [("Gil L18 (published)", "published/gil_L18", "gil_L18"),
        ("ours, L18, margins", "batch/vac-L18-c2", "ours_vac-L18-c2"),
        ("ours, L18, exact", "batch/vac-L18-exact", "ours_vac-L18-exact"),
        ("Wechsung L18 (published)", "published/wechsung_L18", "wechsung_L18"),
        ("ours, L18 Wechsung start, margins", "batch/vac-L18-wech-c2", "ours_vac-L18-wech-c2"),
        ("Gil L20 (published)", "published/gil_L20", "gil_L20"),
        ("Wechsung L20 (published)", "published/wechsung_L20", "wechsung_L20"),
        ("ours, L20, margins", "batch/vac-L20-c3", "ours_vac-L20-c3"),
        ("ours, L20, exact", "batch/vac-L20-exact", "ours_vac-L20-exact"),
        ("Gil L24 (published)", "published/gil_L24", "gil_L24"),
        ("ours, L24 Gil start, exact", "batch/vac-L24-feas2", "ours_vac-L24-feas2"),
        ("Wechsung L24 (published)", "published/wechsung_L24", "wechsung_L24"),
        ("ours, L24 Wechsung start, exact", "batch/vac-L24-wech-feas2", "ours_vac-L24-wech-feas2"),
        ("ours, L24 Wechsung start, exact, order 24", "batch/vac-L24-wech-o24-feas", "ours_vac-L24-wech-o24-feas"),
        ("Gil 3 coils (published)", "published/gil_3coil_L18", "gil_3coil_L18"),
        ("ours, 3 coils, margins", "batch/vac-3coil-c2", "ours_vac-3coil-c2"),
        ("ours, 3 coils, exact", "batch/vac-3coil-exact", "ours_vac-3coil-exact")]


def latest_dir(rel):
    """runs/<rel> or its highest -cN continuation."""
    base = P / "runs" / rel
    conts = [q for q in base.parent.glob(base.name + "-c*") if re.fullmatch(re.escape(base.name) + r"-c\d+", q.name)]
    cands = [base] + sorted(conts, key=lambda q: int(re.search(r"-c(\d+)$", q.name).group(1)))
    cands = [c for c in cands if c.is_dir()]
    return cands[-1] if cands else None


def gil_score(rel):
    d = latest_dir(rel)
    if d is None:
        return {}
    scores = sorted(d.glob("gil_score_step*"), key=lambda p: int(re.search(r"step(\d+)$", p.name).group(1)))
    if not scores:
        return {}
    g = scores[-1]
    out = {}
    for f, keys in (("gil_coil_metrics.json", ("length_per_hfp", "max_kappa", "max_msc", "min_cc", "min_cs", "max_force_MN_per_m_per_turn")),
                    ("gil_metrics.json", ("BdotN_over_B", "BdotN_over_B_own")),
                    ("qs_simsopt.json", ("qs_total_51",))):
        if (g / f).exists():
            r = json.load(open(g / f))
            out.update({k: r[k] for k in keys if k in r})
    out["step"] = g.name
    return out


def eval12(name):
    f = P / "runs/lpqa_vacuum_eval_12x12" / name / "row.json"
    return json.load(open(f))["qa_total"] if f.exists() else None


fmt = lambda v, s="{:.3g}": "–" if v is None else s.format(v)
hdr = "| coil set | L/hfp (m) | κmax (1/m) | MSC (1/m²) | cc (m) | cs (m) | ⟨B·n⟩/⟨B⟩ target | ⟨B·n⟩/⟨B⟩ own LCFS | QS, their pipeline (51) | QS, three-term 12×12 | force (MN/m) |"
lines = [hdr, "|" + "---|" * 11]
for label, rel, ev in ROWS:
    g = gil_score(rel)
    lines.append(f"| {label} | {fmt(g.get('length_per_hfp'), '{:.1f}')} | {fmt(g.get('max_kappa'))} | {fmt(g.get('max_msc'))} | "
                 f"{fmt(g.get('min_cc'))} | {fmt(g.get('min_cs'))} | {fmt(g.get('BdotN_over_B'), '{:.2e}')} | "
                 f"{fmt(g.get('BdotN_over_B_own'), '{:.2e}')} | {fmt(g.get('qs_total_51'), '{:.2e}')} | {fmt(eval12(ev), '{:.2e}')} | "
                 f"{fmt(g.get('max_force_MN_per_m_per_turn'), '{:.2f}')} |")
text = "\n".join(lines)
print(text)
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(text + "\n")
