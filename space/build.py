"""Usage: python space/build.py training_log.json error_analysis.json
Build space/index.html (static Space, Transformers.js) with training metrics embedded from training_log.json."""
import json, sys
log = json.load(open(sys.argv[1]))
hist = log["log_history"]
slim = {
    "train": [{"step": h["step"], "loss": round(h["loss"], 4)} for h in hist if "loss" in h],
    "eval": [{"step": h["step"], "epoch": h["epoch"], "acc": round(h["eval_accuracy"], 4), "f1": round(h["eval_macro_f1"], 4)}
             for h in hist if "eval_loss" in h],
    "test": log["test"], "labels": log["labels"], "cm": log["confusion_matrix"],
    "report": {l: {k: log["report"][l][k] for k in ("precision", "recall", "f1-score", "support")} for l in log["labels"]},
}
slim["analysis"] = json.load(open(sys.argv[2]))
html = open("space/template.html").read().replace("__METRICS_JSON__", json.dumps(slim))
open("space/index.html", "w").write(html)
print("wrote space/index.html", len(html), "bytes")
