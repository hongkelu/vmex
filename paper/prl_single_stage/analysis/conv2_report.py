"""Round-2 convergence report: python analysis/conv2_report.py"""
import json, glob, os
rows = []
for f in glob.glob(os.path.join(os.path.dirname(__file__), "..", "runs/conv2/*/*/row.json")):
    r = json.load(open(f)); tag = f.split("/")[-3]; rows.append((r["name"], tag, r))
for n in sorted({x[0] for x in rows}):
    print(n)
    for name, tag, r in sorted((x for x in rows if x[0] == n), key=lambda x: x[1]):
        print(f"  {tag:18s} QS {r['qa_total']:.4e}  B.n {r['bn']:.2e}  K {r['sheet_current']:.2e}  fsq {r.get('fsqr', float('nan')):.1e}  ns {r.get('ns','?')}  {r['seconds']:6.0f} s")
