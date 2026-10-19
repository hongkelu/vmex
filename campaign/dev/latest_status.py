import glob, json, os, re, sys
os.chdir("/root/autodl-tmp/vmex-qa-cases/prod")
nan = float("nan")
for base in sys.argv[1:]:
    cands = [d for d in glob.glob(base + "*") if os.path.isdir(d) and os.path.exists(d + "/metrics.jsonl") and not re.search(r"(bad|crashed|from-)", d)]
    if not cands:
        print("%-36s no metrics" % base); continue
    d = max(cands, key=lambda p: os.path.getmtime(p + "/metrics.jsonl"))
    r = [json.loads(l) for l in open(d + "/metrics.jsonl")]
    x = r[-1]; done = os.path.exists(d + "/summary.json")
    med = sorted(y["step_seconds"] for y in r[1:])[len(r) // 2] if len(r) > 2 else 0
    print("%-36s step %3d%s target %.2e iota %.3f vac %.3f iota_I %.3f DMerc %+.3f aspect %.2f median %.0fs" % (
        d, x["step"], " done" if done else "", x["qa"], x["min_abs_iota"], x.get("vacuum_iota_min", nan),
        x.get("current_iota", nan), x.get("min_d_merc", nan), x["aspect"], med))
