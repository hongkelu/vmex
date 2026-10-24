"""Report the certificate records of an fbrecord.py run (OUT/records/*.npz), from their own runtime metadata.

usage: python records_report.py OUT [--json FILE]

Per record: kind, round, cost, the share of the cost in each row block (layout from the record), the interface RMS of
each component (B.n, pressure, |K|), the certificate's gain and condition, the gain of a current-only correction at
fixed boundary and of the reduced boundary step, the normal share of the 10 weakest directions and the GN step's normal
motion (from the record's own normal operator, at its own state), and the loss/value change from the previous round.
"""
import json
import sys
from pathlib import Path

import numpy as np


def load(path):
    """A record's metadata and arrays."""
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return meta, {k: data[k] for k in data.files if k != "meta"}


def block(meta, rows, name):
    """The rows of a named row block, or None."""
    for b, start, stop in meta["row_blocks"]:
        if b == name:
            return rows[start:stop]
    return None


def analyze(meta, arr):
    """One record's report."""
    out = dict(sequence=meta["sequence"], kind=meta["kind"], round=meta["round"], reason=meta.get("reason"))
    if "rows" not in arr:
        return out
    r, J = arr["rows"], arr["J"]
    n = arr["x"].size
    cost = float(r @ r)
    out["cost"] = 0.5 * cost
    out["shares"] = {b: float(r[s:e] @ r[s:e]) / cost for b, s, e in meta["row_blocks"]}
    weights = meta["options"]["weights"]
    rms = {}
    for name, w in (("B.n", weights[0]), ("pressure", weights[1])):
        v = block(meta, r, name)
        if v is not None and w:
            rms[name] = float(np.sqrt(v @ v)) / w
    ks = [block(meta, r, k) for k in ("K_x", "K_y", "K_z")]
    if all(k is not None for k in ks) and weights[2]:
        rms["K"] = float(np.sqrt(sum(k @ k for k in ks))) / weights[2]
    out["interface_rms"] = rms
    if meta.get("certificate"):
        out["gain"] = meta["certificate"]["stationarity"]
        out["condition"] = meta["certificate"]["condition"]
    Js = J[:, :n] * arr["x_scale"][None]
    cols = {b: (s, e) for b, s, e in meta["column_blocks"]}
    if "current" in cols:
        s, e = cols["current"]
        Qc, _ = np.linalg.qr(Js[:, s:e])
        gc = Qc.T @ r
        out["current_only_gain"] = float(gc @ gc) / cost
        sb, eb = cols["boundary"]
        Jb = Js[:, sb:eb] - Qc @ (Qc.T @ Js[:, sb:eb])
        rr = r - Qc @ gc
        step = np.linalg.lstsq(Jb, rr, rcond=None)[0]
        res = rr - Jb @ step
        out["reduced_boundary_gain"] = float(rr @ rr - res @ res) / cost
    if "normal" in arr:
        U, s, Vt = np.linalg.svd(Js, full_matrices=False)
        g = U.T @ r
        N = arr["normal"]
        weak = Vt[-10:]
        dn = np.sqrt(np.mean((N @ weak.T) ** 2, axis=0))
        out["weak10_normal_rms_per_unit"] = [float(v) for v in dn]
        full = Vt.T @ (g / s)
        out["gn_step_normal_rms_over_a"] = float(np.sqrt(np.mean((N @ full) ** 2)))
        out["gn_gain_from_weakest_100"] = float(np.sum(g[-100:] ** 2)) / cost
    if "values" in arr:
        out["values"] = [float(v) for v in arr["values"]]
    return out


def main():
    """Run the probe."""
    folder = Path(sys.argv[1])
    paths = sorted((folder / "records").glob("[0-9][0-9][0-9][0-9]_*.npz"))
    reports, previous = [], None
    for path in paths:
        meta, arr = load(path)
        rep = analyze(meta, arr)
        if previous is not None and "values" in rep and "values" in previous:
            a, b = np.array(rep["values"]), np.array(previous["values"])
            rep["values_rel_change"] = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))
        if "values" in rep:
            previous = rep
        reports.append(rep)
        line = f"{rep['sequence']:4d} {rep['kind']:9s} r{rep['round']}"
        if "cost" in rep:
            line += (f" cost {rep['cost']:.4e} gain {rep.get('gain', float('nan')):.3e} cond "
                     f"{rep.get('condition', float('nan')):.2e}")
            line += " " + " ".join(f"{k} {v:.3e}" for k, v in rep["interface_rms"].items())
            if "current_only_gain" in rep:
                line += f" current-only {rep['current_only_gain']:.3f} reduced {rep['reduced_boundary_gain']:.3f}"
            if "gn_step_normal_rms_over_a" in rep:
                line += f" GN normal/a {rep['gn_step_normal_rms_over_a']:.2e}"
            if "values" in rep:
                line += f" loss {rep['values'][0]:.6f}"
            if "values_rel_change" in rep:
                line += f" dvalues {rep['values_rel_change']:.1e}"
        if rep.get("reason"):
            line += f" | {rep['reason'][:120]}"
        print(line)
    status = folder / "status.json"
    if status.exists():
        print("status:", json.loads(status.read_text())["status"])
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(reports, indent=1) + "\n")


if __name__ == "__main__":
    main()
