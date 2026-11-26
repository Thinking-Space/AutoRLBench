import json,sys
m=json.load(open(sys.argv[1] if len(sys.argv)>1 else "s.json"))
ok=m.get("metric_after_fix",0)>=m.get("threshold",0.5)
print(json.dumps({"passed":ok}));sys.exit(0 if ok else 1)
