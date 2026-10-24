"""G1: compare the certified free boundaries of one fixture solved from different starts.

usage: python compare_starts.py OUT.json RUN_DIR [RUN_DIR ...]   (the first run is the reference)

Each run's certified record (records/*_certified.npz) gives the boundary (rbc, zbs), the values (loss and
quantities) and the interface rows.  Reported against the reference: the largest closest-point distance between
the boundary surfaces (16 toroidal cuts of 512 poloidal points, in metres), the largest relative change of any value,
and the RMS of each interface block.  Runs without a certified record are listed as such.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np


def surface(rbc, zbs, nfp, ncut=16, ntheta=512):
    """(R, Z) on ``ncut`` toroidal cuts of one field period."""
    ntor = (rbc.shape[0] - 1) // 2
    m = np.arange(rbc.shape[1])
    n = (np.arange(rbc.shape[0]) - ntor) * nfp
    th = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    cuts = []
    for ph in np.linspace(0, 2 * np.pi / nfp, ncut, endpoint=False):
        a = m[None, None, :] * th[:, None, None] - n[None, :, None] * ph
        cuts.append((np.sum(np.cos(a) * rbc[None], axis=(1, 2)), np.sum(np.sin(a) * zbs[None], axis=(1, 2))))
    return cuts


def distance(c1, c2):
    """Largest distance from a point of one surface to the polygon of the other, over the cuts (both ways)."""
    worst = 0.0
    for (R1, Z1), (R2, Z2) in zip(c1, c2):
        for P, Q in ((np.stack([R1, Z1], 1), np.stack([R2, Z2], 1)), (np.stack([R2, Z2], 1), np.stack([R1, Z1], 1))):
            seg = np.roll(Q, -1, 0) - Q
            t = np.clip(np.einsum("psk,sk->ps", P[:, None] - Q[None], seg) / np.sum(seg**2, 1)[None], 0, 1)
            d = np.min(np.linalg.norm(P[:, None] - (Q[None] + t[..., None] * seg[None]), axis=2), axis=1)
            worst = max(worst, float(d.max()))
    return worst


def load(run):
    """The certified record of ``run``, or None."""
    files = sorted(glob.glob(f"{run}/records/*_certified.npz"))
    if not files:
        return None
    d = np.load(files[-1])
    meta = json.loads(str(d["meta"]))
    rounds = sorted(glob.glob(f"{run}/records/*_round.npz"))
    values = np.load(rounds[-1])["values"] if rounds else None
    return dict(rbc=d["rbc"], zbs=d["zbs"], rows=d["rows"], blocks=meta["row_blocks"], values=values,
                stationarity=meta["certificate"]["stationarity"])


def main():
    """Compare every run with the first; write OUT.json."""
    runs = sys.argv[2:]
    nfp = int(json.loads(Path(runs[0], "config.json").read_text()).get("nfp", 0) or 0) if Path(runs[0], "config.json").exists() else 0
    recs = {r: load(r) for r in runs}
    ref = recs[runs[0]]
    if ref is None:
        raise SystemExit(f"the reference {runs[0]} is not certified")
    if not nfp:
        import re
        text = Path(runs[0], "run", "input.run").read_text() if Path(runs[0], "run", "input.run").exists() else ""
        found = re.search(r"NFP\s*=\s*(\d+)", text, re.I)
        nfp = int(found.group(1)) if found else 1
    c0 = surface(ref["rbc"], ref["zbs"], nfp)
    out = dict(reference=runs[0], nfp=nfp, runs={})
    for r in runs:
        rec = recs[r]
        if rec is None:
            out["runs"][r] = dict(certified=False)
            continue
        row = dict(certified=True, stationarity=rec["stationarity"],
                   surface_distance_m=distance(surface(rec["rbc"], rec["zbs"], nfp), c0))
        if rec["values"] is not None and ref["values"] is not None:
            row["values_max_rel"] = float(np.max(np.abs(rec["values"] - ref["values"])
                                                 / np.maximum(np.abs(ref["values"]), 1e-30)))
        row["blocks_rms"] = {b[0]: float(np.sqrt(np.mean(rec["rows"][b[1]:b[2]] ** 2))) for b in rec["blocks"]}
        out["runs"][r] = row
        print(r, json.dumps({k: v for k, v in row.items() if k != "blocks_rms"}), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
