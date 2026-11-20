---
language: ar
license: cc-by-4.0
library_name: transformers
pipeline_tag: text-classification
base_model: aubmindlab/bert-base-arabertv02
datasets:
- drelhaj/Arabic-Dialects
tags:
- onnx
- transformers.js
- arabic
- dialect-identification
- trackio
metrics:
- accuracy
- f1
model-index:
- name: arabert-arabic-dialect-id
  results:
  - task: {type: text-classification, name: Arabic dialect identification}
    dataset: {name: drelhaj/Arabic-Dialects, type: drelhaj/Arabic-Dialects, split: test}
    metrics:
    - {type: accuracy, value: 0.8026}
    - {type: f1, name: macro F1, value: 0.7511}
    - {type: accuracy, name: accuracy (train overlap removed), value: 0.8008}
    - {type: f1, name: macro F1 (train overlap removed), value: 0.7526}
---

# AraBERT v0.2 fine-tuned for Arabic dialect identification

Classifies an Arabic sentence into one of five varieties: **EGY** (Egyptian), **GLF** (Gulf),
**LAV** (Levantine), **NOR** (North African / Tunisian), **MSA** (Modern Standard Arabic).

This is a learning / practice project (HF NeurIPS 2026 ML Intern Reproductions hackathon prep).

**Code:** training, error-analysis and demo sources are on GitHub at
[pavithrabinu5/arabic-dialect-id](https://github.com/pavithrabinu5/arabic-dialect-id) (MIT license).
The model weights are released under CC-BY-4.0, following the dataset.

## Quick start
```python
from transformers import pipeline
clf = pipeline("text-classification", model="pavithrabinu115/arabert-arabic-dialect-id")
clf("ازيك عامل ايه النهارده؟")
```

## Data
[`drelhaj/Arabic-Dialects`](https://huggingface.co/datasets/drelhaj/Arabic-Dialects) (El-Haj, Rayson & Aboelezz, LREC 2018), using the
provided splits: train 8079, dev 1010, test 1013 sentences (empty rows dropped).
Labels are imbalanced (EGY ≈ 40% of test).
Per the dataset paper, EGY, GLF, LAV and MSA sentences are reader comments from the Arabic Online Commentary dataset
(Zaidan & Callison-Burch, 2014), labelled per sentence by crowd workers; NOR sentences come from the internet-forum section
of the Tunisiya corpus of Tunisian Arabic.

## Method
- Base model: [`aubmindlab/bert-base-arabertv02`](https://huggingface.co/aubmindlab/bert-base-arabertv02) (BERT-base, ~136M params), new 5-way classification head
- Full fine-tuning, 3 epochs, lr 3e-05, linear schedule with 10% warmup, weight decay 0.01,
  batch size 32, max length 128 tokens, fp16, seed 42
- Best checkpoint selected on **dev macro-F1**; numbers below are on the untouched **test** split
- Trained on an Apple Silicon Mac (PyTorch MPS backend)
- Training curves: [Trackio dashboard](https://huggingface.co/spaces/pavithrabinu115/trackio-dashboard) (project `arabic-dialect-id`, run `arabertv02-lr3e-5-ep3`)
- Training script: [`train.py`](./train.py)

## Results (test split)
| Metric | Full test split (n=1013) | Train overlap removed (n=984) |
|---|---|---|
| Accuracy | **0.803** | **0.801** |
| Macro-F1 | **0.751** | **0.753** |
| F1 EGY | 0.993 | 0.992 |
| F1 GLF | 0.634 | 0.636 |
| F1 LAV | 0.600 | 0.600 |
| F1 MSA | 0.862 | 0.862 |
| F1 NOR | 0.667 | 0.673 |

Best dev macro-F1 (used for checkpoint selection): 0.781.

**Train/test overlap.** The test split was searched for sentences that also occur in train: exact matches
(28), matches after normalising the text (diacritics and tatweel removed, alef/ya/ta-marbuta spellings unified, punctuation removed,
stretched letters such as عييييب collapsed; 29 in total), near-duplicates by character 3–5-gram TF-IDF similarity, and
test sentences of 5+ words contained inside a training sentence. The similarity search found nothing beyond the normalised
matches: the 6 extra pairs with cosine ≥ 0.8 are all one- to four-word fragments that merely share a word
(e.g. «منوعات لا لا» vs «منوعات»), none of 5+ words, and there were 0 containment matches. So 29 test sentences (22 EGY, 6 LAV, 1 NOR)
overlap with train, and the right-hand column re-scores the same predictions with them removed. The change is negligible (accuracy 0.803 → 0.801,
macro-F1 0.751 → 0.753): on the 29 overlapping sentences alone accuracy is 0.862, close to the
overall figure, and 7 of them carry a different label in train than in test, so memorisation cannot help there.
They are of two kinds: 22 Egyptian comments duplicated verbatim or with only punctuation changed (some up to 51 words,
some repeated up to four times in train), and 7 one- or two-word items, mostly web-page text scraped with the comments («طباعة» "print", «شارك» "share",
«اجتماعي» "social") or country names, which are labelled differently in train and test.
The dev split has the same issue (29 dev sentences occur in train after the same normalisation); it was only used to pick the epoch.
All other tables below use the full test split.

For reference, the dataset paper reports 76.3% accuracy (SVM with Subtractive Bivalency Profiling features), but with 10-fold
cross-validation on its full 16,494-sentence release. This model is scored once on the 1,013-sentence test split of the
Hugging Face release, so the numbers are not directly comparable.

### Per-dialect results (test split, sorted by F1)
| Dialect | F1 | Precision | Recall | Support | Most often confused with (count, % of that dialect) |
|---|---|---|---|---|---|
| EGY · Egyptian | **0.993** | 0.990 | 0.995 | 407 | almost never (2 errors in 407) |
| MSA · Modern Standard Arabic | **0.862** | 0.797 | 0.940 | 100 | LAV (3, 3%) |
| NOR · North African (Tunisian) | **0.667** | 0.679 | 0.654 | 162 | GLF (27, 17%) |
| GLF · Gulf | **0.634** | 0.619 | 0.649 | 168 | LAV (28, 17%) |
| LAV · Levantine | **0.600** | 0.643 | 0.562 | 176 | GLF (37, 21%) |

### Confusion matrix (test split)
Rows are the true dialect, columns the prediction; percentages are of the row (shown for cells ≥ 5).

| true \ predicted | EGY | GLF | LAV | MSA | NOR | total |
|---|---|---|---|---|---|---|
| **EGY** | **405** | 1 | 1 | 0 | 0 | 407 |
| **GLF** | 1 | **109** | 28 (17%) | 7 (4%) | 23 (14%) | 168 |
| **LAV** | 0 | 37 (21%) | **99** | 14 (8%) | 26 (15%) | 176 |
| **MSA** | 0 | 2 | 3 | **94** | 1 | 100 |
| **NOR** | 3 | 27 (17%) | 23 (14%) | 3 | **106** | 162 |

### Which dialects get confused, and why
**Where the errors are.** The model makes 200 errors on 1013 test sentences.
164 of them (82%) are mix-ups *within* Gulf, Levantine and North African:
GLF↔LAV 65 (37 LAV→GLF, 28 GLF→LAV), GLF↔NOR 50, LAV↔NOR 49.
The next biggest is Levantine predicted as MSA (14). Egyptian is almost never confused (405/407 correct) and MSA has high recall (0.94).

**Why Egyptian is easy and the other three are not.** Counting how often marker words appear in the *training* sentences of
each dialect shows the difference. Egyptian has frequent markers that are close to exclusive to it; the GLF/LAV/NOR markers
are rarer and leak into each other, and the most common dialect words are shared by all three:

| Word | Gloss | EGY | GLF | LAV | MSA | NOR |
|---|---|---|---|---|---|---|
| ايه | what (EGY) | 11.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| عايز | want (EGY) | 3.2 | 0.1 | 0.2 | 0.1 | 0.2 |
| مش | not (EGY, also NOR/LAV) | 23.4 | 3.1 | 6.8 | 0.6 | 11.3 |
| شو | what (LAV, also NOR/GLF) | 0.2 | 1.7 | 4.6 | 0.1 | 2.7 |
| هيك | like this (LAV) | 0.0 | 0.7 | 5.8 | 0.0 | 1.2 |
| بدي | I want (LAV) | 0.0 | 0.2 | 3.8 | 0.1 | 0.8 |
| برشا | a lot (Tunisian) | 0.0 | 0.0 | 0.0 | 0.0 | 1.2 |
| الحين | now (Gulf) | 0.0 | 1.1 | 0.2 | 0.0 | 0.2 |
| يعني | I mean (shared) | 3.0 | 30.1 | 28.7 | 14.0 | 19.3 |
| اللي | which/that (shared) | 9.2 | 19.6 | 17.3 | 1.4 | 17.1 |

*(% of training sentences in each dialect that contain the word.)*

**Short sentences carry little evidence.** Accuracy rises steadily with length. Sentences under 10 words are
29% of the test set but account for 100 of the 200 errors. Some contain no
dialect-specific word at all, e.g. «كما تعرضت لهما» ("as it was exposed to them", labelled GLF, predicted NOR) or
«في العمق» ("in depth", labelled LAV, predicted NOR). The dataset paper makes the same point: one-word
"bivalent" instances are hard even for humans.

| Sentence length (words) | 1-4 | 5-9 | 10-19 | 20+ |
|---|---|---|---|---|
| Accuracy, all dialects | 0.50 | 0.77 | 0.82 | 0.88 |
| Accuracy, excluding EGY | 0.37 | 0.54 | 0.63 | 0.81 |
| Test sentences (non-EGY) | 95 | 83 | 105 | 323 |

For non-Egyptian sentences, the median length of misclassified items is 10 words vs
28 for correct ones.

**Code-switching into MSA.** The texts are reader comments on news articles, and the dataset authors note
that "many writers used a combination of dialect and MSA". Long comments that are mostly MSA with a few dialect words get
pulled towards MSA. For example, a LAV comment that begins «بشكل أولي اتهام سياسي بس هذا الاتهام السياسي مقرونا بالوقائع...» ("initially
a political accusation, but this political accusation, coupled with the facts...") is predicted MSA; «بس» ("but") is its only
clearly dialectal word. The dataset paper also found LAV to be the dialect most often misclassified as MSA.

**Some labels are noisy.** 30 training sentences appear with more than one dialect label, and
7 of the 28 test sentences that also occur word-for-word in the training set carry a *different* label there; these are mostly one-word web-page text such as «طباعة» ("print") or «شارك» ("share"). No model can get
both copies of such a sentence right. (EGY/GLF/LAV/MSA were labelled per sentence by crowd workers; NOR comes from a
separate Tunisian forum corpus.)

**Practical tip: use the confidence score.** Mean top-class probability is 0.91 for correct predictions vs
0.62 for errors. Keeping only predictions with probability ≥ 0.9 covers 66% of the test set at
97.0% accuracy (≥ 0.7: 78% at 90.4%), so low-confidence outputs are best shown as "uncertain".

Full numbers: [`error_analysis.json`](./error_analysis.json), produced by [`error_analysis.py`](./error_analysis.py).

## Browser / ONNX
`onnx/model.onnx` (fp32) and `onnx/model_quantized.onnx` (int8, per-channel dynamic quantization, 136 MB) are included
for [Transformers.js](https://huggingface.co/docs/transformers.js). The int8 model scores the same on the test split
(accuracy 0.803, macro-F1 0.751). A naive per-tensor int8 export dropped to 0.759 / 0.710, so per-channel matters here.
```js
import { pipeline } from "@huggingface/transformers";
const clf = await pipeline("text-classification", "pavithrabinu115/arabert-arabic-dialect-id", { dtype: "q8" });
```
Live demo (runs fully in your browser): [Arabic Dialect ID Space](https://huggingface.co/spaces/pavithrabinu115/arabic-dialect-id)

## Limitations
- Small dataset (~8k training sentences) from a single 2018 corpus; it may not generalise to other domains,
  social media styles, Arabizi (Latin-script Arabic), or dialects not in the label set (e.g. Iraqi, Sudanese, Yemeni).
- "NOR" mixes North African varieties (mostly Tunisian); it should not be read as fine-grained Maghrebi ID.
- Dialect and MSA overlap heavily (bivalency / code-switching); short sentences are often genuinely ambiguous,
  so a single label can be wrong even when the model is "reasonable".
- Inputs are truncated to 128 tokens; no AraBERT-specific text pre-processing was applied.
- Single run, single seed, no hyperparameter search: treat numbers as indicative, not a benchmark result.
- Do not use for decisions about people (e.g. inferring nationality or origin).

## Citation
```bibtex
@inproceedings{elhaj2018arabic,
  title={Arabic Dialect Identification in the Context of Bivalency and Code-Switching},
  author={El-Haj, Mahmoud and Rayson, Paul and Aboelezz, Mariam},
  booktitle={Proceedings of LREC 2018}, pages={3622--3627}, year={2018}
}
```
