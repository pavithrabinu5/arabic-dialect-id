# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers>=5.0",
#   "accelerate",
#   "datasets",
#   "scikit-learn",
#   "trackio",
#   "huggingface_hub",
# ]
# ///
"""Fine-tune AraBERT for 5-way Arabic dialect ID; log to Trackio; push to the Hub.

Run on HF Jobs:
  hf jobs uv run train.py --flavor t4-small --secrets HF_TOKEN -- --push
Run locally (Apple Silicon / CUDA / CPU):
  python train.py --push
"""
import argparse
import json
import os
import tempfile

import numpy as np
import torch
import trackio
from datasets import load_dataset
from huggingface_hub import HfApi
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

DATASET = "drelhaj/Arabic-Dialects"
BASE = "aubmindlab/bert-base-arabertv02"

p = argparse.ArgumentParser()
p.add_argument("--hub-model-id", default="pavithrabinu115/arabert-arabic-dialect-id")
p.add_argument("--trackio-space", default="pavithrabinu115/trackio-dashboard")
p.add_argument("--project", default="arabic-dialect-id")
p.add_argument("--run-name", default="arabertv02-lr3e-5-ep4")
p.add_argument("--epochs", type=float, default=4)
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--max-len", type=int, default=128)
p.add_argument("--max-train-samples", type=int, default=None, help="for smoke tests")
p.add_argument("--push", action="store_true")
args = p.parse_args()

torch.manual_seed(42)
if torch.cuda.is_available():
    HARDWARE = f"a single {torch.cuda.get_device_name(0)} GPU"
elif torch.backends.mps.is_available():
    HARDWARE = "an Apple Silicon Mac (PyTorch MPS backend)"
else:
    HARDWARE = "CPU"
print("Device:", HARDWARE)

# ---------- data ----------
files = {s: f"hf://datasets/{DATASET}/arabic_dialects_{s}.csv" for s in ["train", "dev", "test"]}
ds = load_dataset("csv", data_files=files)
ds = ds.filter(lambda x: isinstance(x["sentence"], str) and len(x["sentence"].strip()) > 0)
labels = sorted(set(ds["train"]["dialect"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
if args.max_train_samples:
    ds["train"] = ds["train"].shuffle(seed=42).select(range(args.max_train_samples))
    ds["dev"] = ds["dev"].select(range(min(200, len(ds["dev"]))))
    ds["test"] = ds["test"].select(range(min(200, len(ds["test"]))))
print({k: len(v) for k, v in ds.items()}, labels)

tok = AutoTokenizer.from_pretrained(BASE)


def prep(b):
    enc = tok(b["sentence"], truncation=True, max_length=args.max_len)
    enc["labels"] = [label2id[d] for d in b["dialect"]]
    return enc


ds = ds.map(prep, batched=True, remove_columns=["sentence", "dialect"])

# ---------- model ----------
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=len(labels), id2label=id2label, label2id=label2id
)


def compute_metrics(ev):
    preds = np.argmax(ev.predictions, axis=-1)
    return {
        "accuracy": accuracy_score(ev.label_ids, preds),
        "macro_f1": f1_score(ev.label_ids, preds, average="macro"),
    }


# ---------- trackio ----------
config = {**vars(args), "base_model": BASE, "dataset": DATASET, "labels": labels}
trackio.init(project=args.project, name=args.run_name, space_id=args.trackio_space or None, config=config)


class TrackioCallback(TrainerCallback):
    def on_log(self, a, state, control, logs=None, **kw):
        if logs:
            m = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            m["epoch"] = state.epoch
            trackio.log(m, step=state.global_step)


targs = TrainingArguments(
    output_dir="out",
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=64,
    warmup_steps=0.1,  # float < 1 = ratio of total steps (transformers 5 API)
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    save_only_model=True,  # skip optimizer state -> much smaller checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=20,
    train_sampling_strategy="group_by_length",  # batch similar lengths together -> less padding, faster
    fp16=torch.cuda.is_available(),
    report_to="none",  # we log to Trackio ourselves via the callback
    seed=42,
)
trainer = Trainer(
    model=model,
    args=targs,
    train_dataset=ds["train"],
    eval_dataset=ds["dev"],
    data_collator=DataCollatorWithPadding(tok),
    processing_class=tok,
    compute_metrics=compute_metrics,
    callbacks=[TrackioCallback()],
)
trainer.train()

# ---------- final test eval ----------
pred = trainer.predict(ds["test"])
yp = np.argmax(pred.predictions, axis=-1)
yt = pred.label_ids
test = {
    "test_accuracy": float(accuracy_score(yt, yp)),
    "test_macro_f1": float(f1_score(yt, yp, average="macro")),
}
report = classification_report(yt, yp, target_names=labels, digits=3, output_dict=True)
cm = confusion_matrix(yt, yp).tolist()
dev = trainer.evaluate()
print("TEST:", test)
print(classification_report(yt, yp, target_names=labels, digits=3))
trackio.log({**test, "dev_macro_f1_best": dev["eval_macro_f1"]}, step=trainer.state.global_step)
trackio.finish()

if not args.push:
    raise SystemExit(0)

# ---------- model card + push ----------
user = args.hub_model_id.split("/")[0]
trackio_url = f"https://huggingface.co/spaces/{args.trackio_space}"
job_id = os.environ.get("JOB_ID", "")
per_class = "\n".join(
    f"| {l} | {report[l]['precision']:.3f} | {report[l]['recall']:.3f} | {report[l]['f1-score']:.3f} | {int(report[l]['support'])} |"
    for l in labels
)
cm_rows = "\n".join(f"| **{labels[i]}** | " + " | ".join(str(c) for c in row) + " |" for i, row in enumerate(cm))
card = f"""---
language: ar
license: cc-by-4.0
library_name: transformers
pipeline_tag: text-classification
base_model: {BASE}
datasets:
- {DATASET}
tags:
- arabic
- dialect-identification
- trackio
metrics:
- accuracy
- f1
model-index:
- name: {args.hub_model_id.split('/')[1]}
  results:
  - task: {{type: text-classification, name: Arabic dialect identification}}
    dataset: {{name: {DATASET}, type: {DATASET}, split: test}}
    metrics:
    - {{type: accuracy, value: {test['test_accuracy']:.4f}}}
    - {{type: f1, name: macro F1, value: {test['test_macro_f1']:.4f}}}
---

# AraBERT v0.2 fine-tuned for Arabic dialect identification

Classifies an Arabic sentence into one of five varieties: **EGY** (Egyptian), **GLF** (Gulf),
**LAV** (Levantine), **NOR** (North African / Tunisian), **MSA** (Modern Standard Arabic).

This is a learning / practice project (HF NeurIPS 2026 ML Intern Reproductions hackathon prep).

## Quick start
```python
from transformers import pipeline
clf = pipeline("text-classification", model="{args.hub_model_id}")
clf("ازيك عامل ايه النهارده؟")
```

## Data
[`{DATASET}`](https://huggingface.co/datasets/{DATASET}) (El-Haj, Rayson & Aboelezz, LREC 2018), using the
provided splits: train {len(ds['train'])}, dev {len(ds['dev'])}, test {len(ds['test'])} sentences (empty rows dropped).
Labels are imbalanced (EGY ≈ 40% of test).

## Method
- Base model: [`{BASE}`](https://huggingface.co/{BASE}) (BERT-base, ~136M params), new 5-way classification head
- Full fine-tuning, {args.epochs:g} epochs, lr {args.lr}, linear schedule with 10% warmup, weight decay 0.01,
  batch size {args.batch_size}, max length {args.max_len} tokens, fp16, seed 42
- Best checkpoint selected on **dev macro-F1**; numbers below are on the untouched **test** split
- Trained on {HARDWARE}{(' via [Hugging Face Jobs](https://huggingface.co/jobs/' + user + '/' + job_id + ')') if job_id else ''}
- Training curves: [Trackio dashboard]({trackio_url}) (project `{args.project}`, run `{args.run_name}`)
- Training script: [`train.py`](./train.py)

## Results (test split)
| Metric | Value |
|---|---|
| Accuracy | **{test['test_accuracy']:.3f}** |
| Macro-F1 | **{test['test_macro_f1']:.3f}** |
| Best dev macro-F1 | {dev['eval_macro_f1']:.3f} |

For reference, the dataset paper reports ~76% accuracy with its Subtractive Bivalency Profiling approach
(different method and evaluation setup, so this is not a like-for-like comparison).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
{per_class}

Confusion matrix (rows = true, columns = predicted):

| | {' | '.join(labels)} |
|---|{'---|' * len(labels)}
{cm_rows}

## Limitations
- Small dataset (~8k training sentences) from a single 2018 corpus; it may not generalise to other domains,
  social media styles, Arabizi (Latin-script Arabic), or dialects not in the label set (e.g. Iraqi, Sudanese, Yemeni).
- "NOR" mixes North African varieties (mostly Tunisian); it should not be read as fine-grained Maghrebi ID.
- Dialect and MSA overlap heavily (bivalency / code-switching); short sentences are often genuinely ambiguous,
  so a single label can be wrong even when the model is "reasonable".
- Inputs are truncated to {args.max_len} tokens; no AraBERT-specific text pre-processing was applied.
- Single run, single seed, no hyperparameter search: treat numbers as indicative, not a benchmark result.
- Do not use for decisions about people (e.g. inferring nationality or origin).

## Citation
```bibtex
@inproceedings{{elhaj2018arabic,
  title={{Arabic Dialect Identification in the Context of Bivalency and Code-Switching}},
  author={{El-Haj, Mahmoud and Rayson, Paul and Aboelezz, Mariam}},
  booktitle={{Proceedings of LREC 2018}}, pages={{3622--3627}}, year={{2018}}
}}
```
"""

out = tempfile.mkdtemp()
trainer.save_model(out)
tok.save_pretrained(out)
with open(os.path.join(out, "README.md"), "w") as f:
    f.write(card)
with open(os.path.join(out, "training_log.json"), "w") as f:
    json.dump({"log_history": trainer.state.log_history, "test": test, "report": report,
               "confusion_matrix": cm, "labels": labels, "config": config}, f, indent=2, default=str)
with open(os.path.join(out, "train.py"), "w") as f:
    f.write(open(__file__).read())

api = HfApi()
api.create_repo(args.hub_model_id, exist_ok=True)
api.upload_folder(folder_path=out, repo_id=args.hub_model_id, commit_message=f"Train {args.run_name}")
print("PUSHED:", f"https://huggingface.co/{args.hub_model_id}")
