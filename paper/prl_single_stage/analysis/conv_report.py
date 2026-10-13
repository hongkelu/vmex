"""Free-boundary convergence report: QS error, B.n and solve time vs boundary tolerance and radial resolution.
    python analysis/conv_report.py"""
import json
from pathlib import Path
P = Path(__file__).resolve().parents[1]
cases = [("12x12 NS50 ftol1e-4 (baseline)", P / "runs/lpqa_vacuum_eval_12x12"), ("NS50 ftol1e-6", P / "runs/bconv_test/f1e-6"),
         ("NS50 ftol1e-8", P / "runs/bconv_test/f1e-8"), ("NS99 ftol1e-8", P / "runs/nsconv_test/ns99"),
         ("NS151 ftol1e-8", P / "runs/nsconv_test/ns151"), ("NS50 b1e-8 fsq1e-15", P / "runs/ftolconv_test/ft1e-15"),
         ("NS50 b1e-8 fsq1e-18", P / "runs/ftolconv_test/ft1e-18"), ("NS50 b1e-8 fsq1e-20", P / "runs/ftolconv_test/ft1e-20")]
for n in ("gilpar_L30_F14", "hurfront_8994ff57", "ours_ff25GA-F10"):
    print(n)
    ref = None
    for lab, d in cases:
        f = d / n / "row.json"
        if not f.exists():
            print(f"  {lab:32s} -"); continue
        r = json.load(open(f)); ref = ref or r["qa_total"]
        print(f"  {lab:32s} QS {r['qa_total']:.4e} ({r['qa_total'] / ref:5.3f})  B.n {r["bn"]:.2e} fsq {r.get("fsqr", float("nan")):.1e}  nfev {r['nfev']:3d}  {r['seconds']:6.0f} s")
