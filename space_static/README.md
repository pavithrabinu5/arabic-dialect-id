---
title: Arabic Dialect ID
emoji: 🗣️
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: cc-by-4.0
models:
- pavithrabinu115/arabert-arabic-dialect-id
datasets:
- drelhaj/Arabic-Dialects
short_description: Classify Arabic text as EGY, GLF, LAV, NOR or MSA
---

In-browser demo of [pavithrabinu115/arabert-arabic-dialect-id](https://huggingface.co/pavithrabinu115/arabert-arabic-dialect-id)
(AraBERT v0.2 fine-tuned on `drelhaj/Arabic-Dialects`). The int8 ONNX model runs client-side with
[Transformers.js](https://huggingface.co/docs/transformers.js), so this is a free static Space with no server.
Training metrics are embedded from the model repo's `training_log.json`; the full run is on the
[Trackio dashboard](https://huggingface.co/spaces/pavithrabinu115/trackio-dashboard).
