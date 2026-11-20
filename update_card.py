"""Usage: python update_card.py  (after train.py --push and error_analysis.py)
Build card/README.md from card/README_base.md + card/training_log.json + error_analysis.json.
Rewrite the Results section of card/README.md with per-dialect table, confusion matrix and error analysis."""
import json, re
A = json.load(open("error_analysis.json")); T = json.load(open("card/training_log.json"))
labs, rep, cm = T["labels"], T["report"], T["confusion_matrix"]
NAMES = {"EGY": "Egyptian", "GLF": "Gulf", "LAV": "Levantine", "NOR": "North African (Tunisian)", "MSA": "Modern Standard Arabic"}
mc = A["most_confused_with"]
def conf_cell(l):
    m = mc[l]
    return "almost never (2 errors in 407)" if l == "EGY" else f"{m['label']} ({m['count']}, {m['share_of_class']:.0%})"
rows = sorted(labs, key=lambda l: -rep[l]["f1-score"])
per = "\n".join(f"| {l} · {NAMES[l]} | **{rep[l]['f1-score']:.3f}** | {rep[l]['precision']:.3f} | {rep[l]['recall']:.3f} | {int(rep[l]['support'])} | {conf_cell(l)} |" for l in rows)
cmrows = "\n".join("| **" + labs[i] + "** | " + " | ".join((f"**{c}**" if i == j else (f"{c} ({c/sum(row):.0%})" if c >= 5 else str(c))) for j, c in enumerate(row)) + f" | {sum(row)} |" for i, row in enumerate(cm))
p = {(x["a"], x["b"]): x for x in A["pairs"]}
gl, gn, ln, lm = p[("GLF", "LAV")], p[("GLF", "NOR")], p[("LAV", "NOR")], p[("LAV", "MSA")]
mk = A["marker_doc_freq_pct_train"]
def mrow(w, gloss): return f"| {w} | {gloss} | " + " | ".join(f"{mk[w][c]:.1f}" for c in labs) + " |"
lb = A["accuracy_by_length"]; th = {t["min_conf"]: t for t in A["confidence"]["thresholds"]}; ln_ = A["label_noise"]
short_share = sum(b["n_all"] for b in lb[:2]) / A["n_test"]
section = f"""### Per-dialect results (test split, sorted by F1)
| Dialect | F1 | Precision | Recall | Support | Most often confused with (count, % of that dialect) |
|---|---|---|---|---|---|
{per}

### Confusion matrix (test split)
Rows are the true dialect, columns the prediction; percentages are of the row (shown for cells ≥ 5).

| true \\ predicted | {' | '.join(labs)} | total |
|---|{'---|' * (len(labs) + 1)}
{cmrows}

### Which dialects get confused, and why
**Where the errors are.** The model makes {A['errors']} errors on {A['n_test']} test sentences.
{A['triangle_GLF_LAV_NOR_errors']} of them ({A['triangle_GLF_LAV_NOR_errors']/A['errors']:.0%}) are mix-ups *within* Gulf, Levantine and North African:
GLF↔LAV {gl['total']} ({gl['b_as_a']} LAV→GLF, {gl['a_as_b']} GLF→LAV), GLF↔NOR {gn['total']}, LAV↔NOR {ln['total']}.
The next biggest is Levantine predicted as MSA ({lm['a_as_b']}). Egyptian is almost never confused (405/407 correct) and MSA has high recall (0.94).

**Why Egyptian is easy and the other three are not.** Counting how often marker words appear in the *training* sentences of
each dialect shows the difference. Egyptian has frequent markers that are close to exclusive to it; the GLF/LAV/NOR markers
are rarer and leak into each other, and the most common dialect words are shared by all three:

| Word | Gloss | {' | '.join(labs)} |
|---|---|{'---|' * len(labs)}
{mrow('ايه', 'what (EGY)')}
{mrow('عايز', 'want (EGY)')}
{mrow('مش', 'not (EGY, also NOR/LAV)')}
{mrow('شو', 'what (LAV, also NOR/GLF)')}
{mrow('هيك', 'like this (LAV)')}
{mrow('بدي', 'I want (LAV)')}
{mrow('برشا', 'a lot (Tunisian)')}
{mrow('الحين', 'now (Gulf)')}
{mrow('يعني', 'I mean (shared)')}
{mrow('اللي', 'which/that (shared)')}

*(% of training sentences in each dialect that contain the word.)*

**Short sentences carry little evidence.** Accuracy rises steadily with length. Sentences under 10 words are
{short_share:.0%} of the test set but account for {A['errors_under_10_words']} of the {A['errors']} errors. Some contain no
dialect-specific word at all, e.g. «كما تعرضت لهما» ("as it was exposed to them", labelled GLF, predicted NOR) or
«في العمق» ("in depth", labelled LAV, predicted NOR). The dataset paper makes the same point: one-word
"bivalent" instances are hard even for humans.

| Sentence length (words) | {' | '.join(b['words'] for b in lb)} |
|---|{'---|' * len(lb)}
| Accuracy, all dialects | {' | '.join(f"{b['acc_all']:.2f}" for b in lb)} |
| Accuracy, excluding EGY | {' | '.join(f"{b['acc_non_egy']:.2f}" for b in lb)} |
| Test sentences (non-EGY) | {' | '.join(str(b['n_non_egy']) for b in lb)} |

For non-Egyptian sentences, the median length of misclassified items is {A['median_words_non_egy']['wrong']:.0f} words vs
{A['median_words_non_egy']['correct']:.0f} for correct ones.

**Code-switching into MSA.** The texts are reader comments on news articles, and the dataset authors note
that "many writers used a combination of dialect and MSA". Long comments that are mostly MSA with a few dialect words get
pulled towards MSA. For example, a LAV comment that begins «بشكل أولي اتهام سياسي بس هذا الاتهام السياسي مقرونا بالوقائع...» ("initially
a political accusation, but this political accusation, coupled with the facts...") is predicted MSA; «بس» ("but") is its only
clearly dialectal word. The dataset paper also found LAV to be the dialect most often misclassified as MSA.

**Some labels are noisy.** {ln_['train_sentences_with_multiple_labels']} training sentences appear with more than one dialect label, and
{ln_['test_in_train_label_conflict']} of the {ln_['test_sentences_in_train']} test sentences that also occur word-for-word in the training set carry a *different* label there; these are mostly one-word web-page text such as «طباعة» ("print") or «شارك» ("share"). No model can get
both copies of such a sentence right. (EGY/GLF/LAV/MSA were labelled per sentence by crowd workers; NOR comes from a
separate Tunisian forum corpus.)

**Practical tip: use the confidence score.** Mean top-class probability is {A['confidence']['mean_correct']:.2f} for correct predictions vs
{A['confidence']['mean_wrong']:.2f} for errors. Keeping only predictions with probability ≥ 0.9 covers {th[0.9]['coverage']:.0%} of the test set at
{th[0.9]['accuracy']:.1%} accuracy (≥ 0.7: {th[0.7]['coverage']:.0%} at {th[0.7]['accuracy']:.1%}), so low-confidence outputs are best shown as "uncertain".

Full numbers: [`error_analysis.json`](./error_analysis.json), produced by [`error_analysis.py`](./error_analysis.py).

"""
s = open("card/README_base.md").read()  # card as written by train.py; this script adds the analysis sections
start = s.index("| Class | Precision | Recall | F1 | Support |")
end = s.index("## Browser / ONNX")
s = s[:start] + section + s[end:]
O = A["overlap_removed"]; F, D, OO = O["full"], O["deduplicated"], O["on_overlap_only"]
rb = ", ".join(f"{v} {k}" for k, v in O["removed_by_dialect"].items() if v)
old_tbl = s[s.index("| Metric | Value |"):s.index("For reference, the dataset paper")]
new_tbl = f"""| Metric | Full test split (n={A['n_test']}) | Train overlap removed (n={O['n_kept']}) |
|---|---|---|
| Accuracy | **{F['accuracy']:.3f}** | **{D['accuracy']:.3f}** |
| Macro-F1 | **{F['macro_f1']:.3f}** | **{D['macro_f1']:.3f}** |
""" + "\n".join(f"| F1 {l} | {rep[l]['f1-score']:.3f} | {D['per_class_f1'][l]:.3f} |" for l in labs) + f"""

Best dev macro-F1 (used for checkpoint selection): {max(h['eval_macro_f1'] for h in T['log_history'] if 'eval_macro_f1' in h):.3f}.

**Train/test overlap.** The test split was searched for sentences that also occur in train: exact matches
({O['checks']['exact']}), matches after normalising the text (diacritics and tatweel removed, alef/ya/ta-marbuta spellings unified, punctuation removed,
stretched letters such as عييييب collapsed; {O['checks']['normalized_exact']} in total), near-duplicates by character 3–5-gram TF-IDF similarity, and
test sentences of 5+ words contained inside a training sentence. The similarity search found nothing beyond the normalised
matches: the {O['checks']['fuzzy_cos_ge_0.8_beyond_normalized']} extra pairs with cosine ≥ 0.8 are all one- to four-word fragments that merely share a word
(e.g. «منوعات لا لا» vs «منوعات»), none of 5+ words, and there were {O['checks']['test_contained_in_train_5plus_words']} containment matches. So {O['n_removed']} test sentences ({rb})
overlap with train, and the right-hand column re-scores the same predictions with them removed. The change is negligible (accuracy {F['accuracy']:.3f} → {D['accuracy']:.3f},
macro-F1 {F['macro_f1']:.3f} → {D['macro_f1']:.3f}): on the {O['n_removed']} overlapping sentences alone accuracy is {OO['accuracy']:.3f}, close to the
overall figure, and {OO['label_conflict_with_train']} of them carry a different label in train than in test, so memorisation cannot help there.
They are of two kinds: {O['removed_by_dialect']['EGY']} Egyptian comments duplicated verbatim or with only punctuation changed (some up to 51 words,
some repeated up to four times in train), and {OO['label_conflict_with_train']} one- or two-word items, mostly web-page text scraped with the comments («طباعة» "print", «شارك» "share",
«اجتماعي» "social") or country names, which are labelled differently in train and test.
The dev split has the same issue ({O['dev_sentences_in_train']} dev sentences occur in train after the same normalisation); it was only used to pick the epoch.
All other tables below use the full test split.

"""
s = s.replace(old_tbl, new_tbl)
s = s.replace("""For reference, the dataset paper reports ~76% accuracy with its Subtractive Bivalency Profiling approach
(different method and evaluation setup, so this is not a like-for-like comparison).""",
"""For reference, the dataset paper reports 76.3% accuracy (SVM with Subtractive Bivalency Profiling features), but with 10-fold
cross-validation on its full 16,494-sentence release. This model is scored once on the 1,013-sentence test split of the
Hugging Face release, so the numbers are not directly comparable.""")
s = s.replace("""Labels are imbalanced (EGY ≈ 40% of test).""",
"""Labels are imbalanced (EGY ≈ 40% of test).
Per the dataset paper, EGY, GLF, LAV and MSA sentences are reader comments from the Arabic Online Commentary dataset
(Zaidan & Callison-Burch, 2014), labelled per sentence by crowd workers; NOR sentences come from the internet-forum section
of the Tunisiya corpus of Tunisian Arabic.""")
s = s.replace("""    - {type: f1, name: macro F1, value: 0.7511}""", f"""    - {{type: f1, name: macro F1, value: 0.7511}}
    - {{type: accuracy, name: accuracy (train overlap removed), value: {D['accuracy']:.4f}}}
    - {{type: f1, name: macro F1 (train overlap removed), value: {D['macro_f1']:.4f}}}""")
assert "train overlap removed" in s.split("---")[1], "model-index not updated"
assert "Which dialects get confused" in s and "16,494" in s and "Tunisiya" in s
open("card/README.md", "w").write(s)
print("ok", len(s))
