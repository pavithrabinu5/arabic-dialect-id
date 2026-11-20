"""Post-hoc error analysis of the dialect-ID model on the test split -> error_analysis.json."""
import json, re, collections, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
M = "pavithrabinu115/arabert-arabic-dialect-id"
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForSequenceClassification.from_pretrained(M).to(dev).eval()
files = {s: f"hf://datasets/drelhaj/Arabic-Dialects/arabic_dialects_{s}.csv" for s in ["train", "test"]}
ds = load_dataset("csv", data_files=files).filter(lambda x: isinstance(x["sentence"], str) and x["sentence"].strip())
te, tr = ds["test"], ds["train"]
labs = [model.config.id2label[i] for i in range(model.config.num_labels)]; l2i = {l: i for i, l in enumerate(labs)}
P = []
with torch.no_grad():
    for i in range(0, len(te), 32):
        e = tok(te["sentence"][i:i+32], truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)
        P.append(torch.softmax(model(**e).logits, -1).cpu().numpy())
P = np.concatenate(P); yp = P.argmax(1); yt = np.array([l2i[d] for d in te["dialect"]])
nw = np.array([len(s.split()) for s in te["sentence"]]); conf = P.max(1); wrong = yp != yt
cm = np.zeros((5, 5), int)
for a, b in zip(yt, yp): cm[a, b] += 1
pairs = sorted([{"a": labs[i], "b": labs[j], "total": int(cm[i, j] + cm[j, i]), "a_as_b": int(cm[i, j]), "b_as_a": int(cm[j, i])}
                for i in range(5) for j in range(i + 1, 5)], key=lambda p: -p["total"])
top_conf = {}
for i, l in enumerate(labs):
    off = [(cm[i, j], labs[j]) for j in range(5) if j != i]; c, o = max(off)
    top_conf[l] = {"label": o, "count": int(c), "share_of_class": round(c / cm[i].sum(), 3)}
ne = yt != l2i["EGY"]; buckets = []
for lo, hi, name in [(0, 5, "1-4"), (5, 10, "5-9"), (10, 20, "10-19"), (20, 10**9, "20+")]:
    m = (nw >= lo) & (nw < hi)
    buckets.append({"words": name, "n_all": int(m.sum()), "acc_all": round(float((~wrong[m]).mean()), 3),
                    "n_non_egy": int((m & ne).sum()), "acc_non_egy": round(float((~wrong[m & ne]).mean()), 3)})
tri = {"GLF", "LAV", "NOR"}
tri_err = sum(p["total"] for p in pairs if {p["a"], p["b"]} <= tri)
def words(s): return set(re.findall(r"[؀-ۿ]+", s))
df = {c: collections.Counter() for c in labs}; n = collections.Counter(tr["dialect"])
for s, d in zip(tr["sentence"], tr["dialect"]): df[d].update(words(s))
markers = ["ايه", "عايز", "مش", "شو", "هيك", "بدي", "برشا", "الحين", "يعني", "اللي"]
trs = collections.defaultdict(set)
for s, d in zip(tr["sentence"], tr["dialect"]): trs[s.strip()].add(d)
ov = [(s, d) for s, d in zip(te["sentence"], te["dialect"]) if s.strip() in trs]
out = {
    "n_test": int(len(te)), "errors": int(wrong.sum()), "labels": labs, "pairs": pairs, "most_confused_with": top_conf,
    "triangle_GLF_LAV_NOR_errors": int(tri_err),
    "errors_under_10_words": int((wrong & (nw < 10)).sum()),
    "median_words_non_egy": {"correct": float(np.median(nw[ne & ~wrong])), "wrong": float(np.median(nw[ne & wrong]))},
    "accuracy_by_length": buckets,
    "confidence": {"mean_correct": round(float(conf[~wrong].mean()), 3), "mean_wrong": round(float(conf[wrong].mean()), 3),
                   "thresholds": [{"min_conf": t, "coverage": round(float((conf >= t).mean()), 3),
                                   "accuracy": round(float((~wrong[conf >= t]).mean()), 3)} for t in (0.5, 0.7, 0.9)]},
    "marker_doc_freq_pct_train": {w: {c: round(100 * df[c][w] / n[c], 1) for c in labs} for w in markers},
    "label_noise": {"train_sentences_with_multiple_labels": sum(len(v) > 1 for v in trs.values()),
                    "test_sentences_in_train": len(ov), "test_in_train_label_conflict": sum(d not in trs[s.strip()] for s, d in ov)},
}
# ---- Train/test overlap: exact, normalized-exact, fuzzy (char n-gram TF-IDF) and containment checks ----
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
def norm(s):
    s = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", s)              # diacritics, tatweel
    s = re.sub(r"[إأآٱ]", "ا", s); s = s.replace("ى", "ي").replace("ة", "ه")
    s = re.sub(r"[^\w\s]|_", " ", s); s = re.sub(r"(.)\1{2,}", r"\1", s)  # punctuation; letters stretched 3+ times
    return re.sub(r"\s+", " ", s).strip().lower()
trn = [norm(s) for s in tr["sentence"]]; ten = [norm(s) for s in te["sentence"]]; nset = set(trn)
exact = np.array([s.strip() in trs for s in te["sentence"]])
in_tr = np.array([s in nset for s in ten]); keep = ~in_tr            # leakage set = normalized-exact matches
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(trn + ten)
best_sim = (vec.transform(ten) @ vec.transform(trn).T).toarray().max(1)
big = " " + " ".join(trn) + " "
contain = sum(1 for i, x in enumerate(ten) if not in_tr[i] and len(x.split()) >= 5 and f" {x} " in big)
overlap_checks = {"exact": int(exact.sum()), "normalized_exact": int(in_tr.sum()),
                  "fuzzy_cos_ge_0.9_beyond_normalized": int(((best_sim >= 0.9) & ~in_tr).sum()),
                  "fuzzy_cos_ge_0.8_beyond_normalized": int(((best_sim >= 0.8) & ~in_tr).sum()),
                  "fuzzy_cos_ge_0.8_beyond_normalized_5plus_words": int(((best_sim >= 0.8) & ~in_tr & (nw >= 5)).sum()),
                  "test_contained_in_train_5plus_words": int(contain)}
print("overlap checks:", overlap_checks)
rep_k = classification_report(yt[keep], yp[keep], labels=list(range(5)), target_names=labs, output_dict=True, zero_division=0)
out["overlap_removed"] = {
    "definition": "test sentence equals a train sentence after normalization (diacritics/tatweel removed, alef/ya/ta-marbuta unified, punctuation removed, letters repeated 3+ times collapsed)",
    "checks": overlap_checks,
    "n_removed": int(in_tr.sum()), "n_kept": int(keep.sum()),
    "removed_by_dialect": {c: int((in_tr & (yt == l2i[c])).sum()) for c in labs},
    "full": {"accuracy": round(float(accuracy_score(yt, yp)), 4), "macro_f1": round(float(f1_score(yt, yp, average="macro")), 4)},
    "deduplicated": {"accuracy": round(float(accuracy_score(yt[keep], yp[keep])), 4),
                     "macro_f1": round(float(f1_score(yt[keep], yp[keep], average="macro")), 4),
                     "per_class_f1": {c: round(rep_k[c]["f1-score"], 4) for c in labs}},
    "on_overlap_only": {"accuracy": round(float(accuracy_score(yt[in_tr], yp[in_tr])), 4),
                        "label_conflict_with_train": int(sum(d not in trs[s.strip()] for s, d, m in zip(te["sentence"], te["dialect"], exact) if m)),
                        "median_words": float(np.median(nw[in_tr]))},
    "dev_sentences_in_train": int(sum(norm(s) in nset for s in load_dataset("csv", data_files={"dev": f"hf://datasets/drelhaj/Arabic-Dialects/arabic_dialects_dev.csv"})["dev"].filter(lambda x: isinstance(x["sentence"], str) and x["sentence"].strip())["sentence"])),
}
json.dump(out, open("error_analysis.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ["overlap_removed", "errors", "triangle_GLF_LAV_NOR_errors", "errors_under_10_words", "most_confused_with", "median_words_non_egy", "label_noise"]}, ensure_ascii=False, indent=1))
