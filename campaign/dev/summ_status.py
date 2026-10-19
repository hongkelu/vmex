import json, os
os.chdir("/root/autodl-tmp")
runs = [("free baseline w=0", "vmex-opt-c/prod/free-qa4-beta"), ("free jbs w=0.01", "vmex-qa-cases/prod/free-qa4-beta-jbs"),
        ("free jbs w=0.1", "vmex-qa-cases/prod/free-qa4-beta-jbs-w0.1"), ("free jbs w=1", "vmex-qa-cases/prod/free-qa4-beta-jbs-w1"),
        ("free merc s>=0.2", "vmex-qa-cases/prod/free-qa4-beta-merc-s02-init"), ("fixed baseline", "vmex-opt-c/prod/fixed-qa4-beta"),
        ("fixed jbs w=0.01", "vmex-qa-cases/prod/fixed-qa4-beta-jbs"), ("fixed merc s>=0.2", "vmex-qa-cases/prod/fixed-qa4-beta-merc-s02-init"),
        ("free QI", "vmex-opt-c/prod/free-qi6-beta"), ("fixed QI", "vmex-opt-c/prod/fixed-qi6-beta")]
nan = float("nan")
for name, d in runs:
    p = d + "/metrics.jsonl"
    if not os.path.exists(p):
        print(name, "- no metrics yet"); continue
    r = [json.loads(l) for l in open(p)]
    x = r[-1]; done = " done" if os.path.exists(d + "/summary.json") else ""
    med = sorted(y["step_seconds"] for y in r[1:])[len(r) // 2] if len(r) > 2 else 0
    def at(k):
        m = [y for y in r if y["step"] == k]
        return "%.1e" % m[0]["qa"] if m else "-"
    print("%-18s step %3d%-5s target %.2e (s50 %s, s100 %s) iota_I %.3f DMerc %.3f vac_iota %.3f iota %.3f median %.0fs" % (
        name, x["step"], done, x["qa"], at(50), at(100), x.get("current_iota", nan), x.get("min_d_merc", nan),
        x.get("vacuum_iota_min", nan), x["min_abs_iota"], med))
