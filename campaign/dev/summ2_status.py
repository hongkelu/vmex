import json, os, sys
os.chdir("/root/autodl-tmp")
runs = [l.split("|") for l in sys.argv[1:]]
nan = float("nan")
for name, d in runs:
    p = d + "/metrics.jsonl"
    if not os.path.exists(p):
        print("%-22s no metrics yet" % name); continue
    r = [json.loads(l) for l in open(p)]
    x = r[-1]; done = " done" if os.path.exists(d + "/summary.json") else ""
    alive = os.popen("pgrep -f '[o]utput %s' | head -1" % d.split("/", 1)[-1]).read().strip()
    med = sorted(y["step_seconds"] for y in r[1:])[len(r) // 2] if len(r) > 2 else 0
    print("%-22s step %3d%-5s target %.2e  iota %.3f vac %.3f iota_I %.3f DMerc %+.3f aspect %.2f median %.0fs %s" % (
        name, x["step"], done, x["qa"], x["min_abs_iota"], x.get("vacuum_iota_min", nan), x.get("current_iota", nan),
        x.get("min_d_merc", nan), x["aspect"], med, "" if done or alive else "(not running)"))
