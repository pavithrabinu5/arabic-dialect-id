# Arabic Dialect ID

Fine-tunes [AraBERT v0.2](https://huggingface.co/aubmindlab/bert-base-arabertv02) (136M parameters) to classify an Arabic
sentence as one of five varieties: **EGY** (Egyptian), **GLF** (Gulf), **LAV** (Levantine), **NOR** (North African / Tunisian)
or **MSA** (Modern Standard Arabic). Training is logged with [Trackio](https://github.com/gradio-app/trackio), the model is
published on the Hugging Face Hub, and a demo runs entirely in the browser with Transformers.js.

This was a practice run for the Hugging Face NeurIPS 2026 ML Intern Reproductions hackathon.

| | Link |
|---|---|
| 🤗 Model (with full model card) | https://huggingface.co/pavithrabinu115/arabert-arabic-dialect-id |
| 🗣️ Demo Space (runs in your browser) | https://huggingface.co/spaces/pavithrabinu115/arabic-dialect-id |
| 📈 Trackio dashboard | https://huggingface.co/spaces/pavithrabinu115/trackio-dashboard |
| 📚 Dataset | [`drelhaj/Arabic-Dialects`](https://huggingface.co/datasets/drelhaj/Arabic-Dialects) (El-Haj, Rayson & Aboelezz, LREC 2018) |

## Results

Test split of `drelhaj/Arabic-Dialects`. The right-hand column removes the 29 test sentences that also occur in the training
split (see [Train/test overlap](#traintest-overlap)).

| Metric | Full test split (n=1013) | Train overlap removed (n=984) |
|---|---|---|
| Accuracy | **0.803** | **0.801** |
| Macro-F1 | **0.751** | **0.753** |
| F1 EGY | 0.993 | 0.992 |
| F1 MSA | 0.862 | 0.862 |
| F1 NOR | 0.667 | 0.673 |
| F1 GLF | 0.634 | 0.636 |
| F1 LAV | 0.600 | 0.600 |

Best dev macro-F1 (used to pick the epoch): 0.781. Single run, single seed, no hyperparameter search.
The dataset paper reports 76.3% accuracy, but with 10-fold cross-validation on its full 16,494-sentence release, so the
numbers are not directly comparable.

### Confusion matrix (full test split)
| true \ predicted | EGY | GLF | LAV | MSA | NOR |
|---|---|---|---|---|---|
| **EGY** | **405** | 1 | 1 | 0 | 0 |
| **GLF** | 1 | **109** | 28 | 7 | 23 |
| **LAV** | 0 | 37 | **99** | 14 | 26 |
| **MSA** | 0 | 2 | 3 | **94** | 1 |
| **NOR** | 3 | 27 | 23 | 3 | **106** |

## Which dialects get confused, and why

- **Most errors are Gulf, Levantine and North African mistaken for each other.** 164 of the 200 test errors (82%):
  GLF↔LAV 65, GLF↔NOR 50, LAV↔NOR 49. Next is Levantine predicted as MSA (14). Egyptian is almost never wrong (405/407).
- **Egyptian has frequent, distinctive words; the other three share theirs.** In training, «ايه» ("what") appears in 11% of
  Egyptian sentences and essentially nowhere else. Levantine «شو» ("what") also shows up in North African (2.7%) and Gulf
  (1.7%) sentences, and the most common dialect words, «يعني» ("I mean") and «اللي» ("which"), occur at similar rates in all three.
- **Short sentences carry little evidence.** Sentences under 10 words are 29% of the test set but cause half of the errors.
  Excluding Egyptian, accuracy is 0.37 for 1–4 words, 0.54 for 5–9, 0.63 for 10–19 and 0.81 for 20+. Some contain no
  dialect word at all, e.g. «في العمق» ("in depth").
- **Code-switching pulls dialect towards MSA.** The texts are reader comments on news articles, and writers often mix
  dialect with MSA. Long, mostly-MSA comments with a word or two of dialect get predicted as MSA; this hits Levantine most,
  as the dataset paper also found.
- **Some labels are noisy.** 30 training sentences appear with more than one label, and 7 test sentences that also occur
  in training carry a different label there; those 7 are mostly one-word web-page text scraped with the comments
  (e.g. «طباعة» "print", «شارك» "share").
- **Confidence is informative.** Predictions with probability ≥ 0.9 cover 66% of the test set at 97.0% accuracy.

Full tables (per-dialect marker-word frequencies, accuracy by sentence length) are in the
[model card](https://huggingface.co/pavithrabinu115/arabert-arabic-dialect-id#which-dialects-get-confused-and-why) and
[`error_analysis.json`](error_analysis.json).

## Train/test overlap

The test split was checked against train with four methods: exact match (28 sentences); match after normalising the text
(diacritics, tatweel, alef/ya/ta-marbuta spellings, punctuation, stretched letters; 29 in total); character 3–5-gram TF-IDF
similarity (no further duplicates: the 6 extra pairs with cosine ≥ 0.8 are short fragments sharing one word); and containment
of 5+-word test sentences inside training sentences (0). The 29 are 22 Egyptian comments copied verbatim or with only
punctuation changed, plus 7 one- or two-word items with conflicting labels. Removing them changes accuracy by −0.2 points
and macro-F1 by +0.2 points, so this overlap does not inflate the results. The dev split has a similar overlap (29 sentences);
it was only used to pick the epoch.

## Repository layout

| Path | What it is |
|---|---|
| `train.py` | Fine-tuning script: loads the data, trains with the HF `Trainer`, logs to Trackio, evaluates on test, writes the model card and pushes to the Hub. Also a [uv script](https://docs.astral.sh/uv/guides/scripts/) for HF Jobs. |
| `error_analysis.py` | Re-runs the published model on the test set: confusion pairs, accuracy by length, marker words, label noise, overlap checks → `error_analysis.json`. |
| `update_card.py` | Builds `card/README.md` (the published model card) from `card/README_base.md`, `card/training_log.json` and `error_analysis.json`. |
| `card/` | Model card sources and the training log downloaded from the model repo. |
| `space_static/` | The published demo Space (static HTML + Transformers.js). |
| `space/` | Demo sources: `template.html` + `build.py` generate `index.html`; `app.py` is a Gradio version for hardware that supports Gradio Spaces. |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf auth login                      # needs a write token

# 1. Train (~18 min on an Apple M4; uses CUDA or MPS if available). Use your own namespace:
python train.py --push \
  --hub-model-id <you>/arabert-arabic-dialect-id \
  --trackio-space <you>/trackio-dashboard
# Without HF PRO, Trackio cannot create a live (Gradio) dashboard Space and logs locally instead.
# Publish a free read-only static dashboard afterwards:
trackio sync --project arabic-dialect-id --space-id <you>/trackio-dashboard --sdk static

# 2. Error analysis (set M at the top of the script to your model id)
python error_analysis.py

# 3. Rebuild and upload the model card
hf download <you>/arabert-arabic-dialect-id README.md training_log.json --local-dir card
mv card/README.md card/README_base.md
python update_card.py
hf upload <you>/arabert-arabic-dialect-id card/README.md README.md
hf upload <you>/arabert-arabic-dialect-id . . --include error_analysis.json --include error_analysis.py

# 4. Rebuild the demo page
python space/build.py card/training_log.json error_analysis.json
```

Quick smoke test before a full run: `python train.py --max-train-samples 200 --epochs 1 --trackio-space ""`.

### Running on Hugging Face Jobs

`train.py` declares its dependencies inline, so it runs on HF infrastructure without a Dockerfile (paid; about $0.10 on a T4):

```bash
hf jobs uv run train.py --flavor t4-small --timeout 45m --secrets HF_TOKEN -- --push
hf jobs logs <job-id> --follow
```

### Browser (ONNX) model

The demo uses `onnx/model_quantized.onnx` from the model repo, exported with `torch.onnx.export` and quantised with
`onnxruntime.quantization.quantize_dynamic(..., weight_type=QuantType.QInt8, per_channel=True)`. Per-channel quantisation
keeps the test scores identical to the full model; per-tensor int8 dropped accuracy to 0.759.

## Limitations

Small corpus (about 8k training sentences) from a single 2018 source of news comments plus Tunisian forums; "NOR" is
mostly Tunisian; dialects not in the label set (e.g. Iraqi, Sudanese, Yemeni) and Arabizi are not covered; short
sentences are often genuinely ambiguous. Do not use for decisions about people (e.g. inferring nationality or origin).
See the model card for details.

## Citation

```bibtex
@inproceedings{elhaj2018arabic,
  title={Arabic Dialect Identification in the Context of Bivalency and Code-Switching},
  author={El-Haj, Mahmoud and Rayson, Paul and Aboelezz, Mariam},
  booktitle={Proceedings of LREC 2018}, pages={3622--3627}, year={2018}
}
```
