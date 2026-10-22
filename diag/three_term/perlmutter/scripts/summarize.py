"""Summarize [trajectory]/[cert] lines of probe logs: per run, rounds, cost, B.n, loss per round, and the loss and
constraint drift over the last rounds.  usage: python summarize.py LOG..."""
import re
import sys

for path in sys.argv[1:]:
    traj, certs = [], []
    for line in open(path, errors="replace"):
        if line.startswith("[trajectory] cost"):
            m = re.search(r"cost (\S+) B\.n (\S+) K (\S+) loss (\S+?)(;|$)", line.strip())
            ch = re.search(r"values rel change (\S+)", line)
            traj.append((float(m.group(1)), float(m.group(2)), float(m.group(4)), float(ch.group(1)) if ch else None))
        elif line.startswith("[cert"):
            m = re.search(r"stationarity (\S+) .*condition (\S+)", line)
            certs.append((float(m.group(1)), float(m.group(2))))
    name = path.split("/")[-1].removesuffix(".log")
    if not traj:
        print(f"{name:16s} no rounds yet")
        continue
    losses = [t[2] for t in traj]
    tail = losses[-4:]
    drift = (max(tail) - min(tail)) / abs(tail[-1]) if len(tail) > 1 else float("nan")
    last_changes = [t[3] for t in traj[-3:] if t[3] is not None]
    print(f"{name:16s} rounds {len(traj):2d}  cost {traj[0][0]:.2e}->{traj[-1][0]:.2e}  B.n {traj[-1][1]:.2e}  "
          f"gain {certs[-1][0] if certs else float('nan'):.3f} cond {certs[-1][1] if certs else float('nan'):.1e}  "
          f"loss last {losses[-1]:.5f}  loss spread (last 4) {drift:.2e}  "
          f"values change/round (last 3) {max(last_changes) if last_changes else float('nan'):.1e}")
