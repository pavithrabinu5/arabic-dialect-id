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
---

# AraBERT v0.2 fine-tuned for Arabic dialect identification

Classifies an Arabic sentence into one of five varieties: **EGY** (Egyptian), **GLF** (Gulf),
**LAV** (Levantine), **NOR** (North African / Tunisian), **MSA** (Modern Standard Arabic).

This is a learning / practice project (HF NeurIPS 2026 ML Intern Reproductions hackathon prep).

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

## Method
- Base model: [`aubmindlab/bert-base-arabertv02`](https://huggingface.co/aubmindlab/bert-base-arabertv02) (BERT-base, ~136M params), new 5-way classification head
- Full fine-tuning, 3 epochs, lr 3e-05, linear schedule with 10% warmup, weight decay 0.01,
  batch size 32, max length 128 tokens, fp16, seed 42
- Best checkpoint selected on **dev macro-F1**; numbers below are on the untouched **test** split
- Trained on an Apple Silicon Mac (PyTorch MPS backend)
- Training curves: [Trackio dashboard](https://huggingface.co/spaces/pavithrabinu115/trackio-dashboard) (project `arabic-dialect-id`, run `arabertv02-lr3e-5-ep3`)
- Training script: [`train.py`](./train.py)

## Results (test split)
| Metric | Value |
|---|---|
| Accuracy | **0.803** |
| Macro-F1 | **0.751** |
| Best dev macro-F1 | 0.781 |

For reference, the dataset paper reports ~76% accuracy with its Subtractive Bivalency Profiling approach
(different method and evaluation setup, so this is not a like-for-like comparison).

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| EGY | 0.990 | 0.995 | 0.993 | 407 |
| GLF | 0.619 | 0.649 | 0.634 | 168 |
| LAV | 0.643 | 0.562 | 0.600 | 176 |
| MSA | 0.797 | 0.940 | 0.862 | 100 |
| NOR | 0.679 | 0.654 | 0.667 | 162 |

Confusion matrix (rows = true, columns = predicted):

| | EGY | GLF | LAV | MSA | NOR |
|---|---|---|---|---|---|
| **EGY** | 405 | 1 | 1 | 0 | 0 |
| **GLF** | 1 | 109 | 28 | 7 | 23 |
| **LAV** | 0 | 37 | 99 | 14 | 26 |
| **MSA** | 0 | 2 | 3 | 94 | 1 |
| **NOR** | 3 | 27 | 23 | 3 | 106 |

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
