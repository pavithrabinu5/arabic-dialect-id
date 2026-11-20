import json

import gradio as gr
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import pipeline

MODEL_ID = "pavithrabinu115/arabert-arabic-dialect-id"
TRACKIO_URL = "https://huggingface.co/spaces/pavithrabinu115/trackio-dashboard"
NAMES = {
    "EGY": "Egyptian",
    "GLF": "Gulf",
    "LAV": "Levantine",
    "NOR": "North African / Tunisian",
    "MSA": "Modern Standard Arabic",
}

clf = pipeline("text-classification", model=MODEL_ID, top_k=None, truncation=True, max_length=128)
log = json.load(open(hf_hub_download(MODEL_ID, "training_log.json")))


def predict(text):
    if not text or not text.strip():
        return {}
    return {f"{NAMES.get(p['label'], p['label'])} ({p['label']})": p["score"] for p in clf(text)[0]}


hist = log["log_history"]
train_df = pd.DataFrame([{"step": h["step"], "loss": h["loss"]} for h in hist if "loss" in h])
eval_df = pd.DataFrame(
    [r for h in hist if "eval_loss" in h for r in (
        {"step": h["step"], "metric": "dev accuracy", "value": h["eval_accuracy"]},
        {"step": h["step"], "metric": "dev macro-F1", "value": h["eval_macro_f1"]},
    )]
)
rep = log["report"]
labels = log["labels"]
class_df = pd.DataFrame(
    [{"class": l, "precision": round(rep[l]["precision"], 3), "recall": round(rep[l]["recall"], 3),
      "f1": round(rep[l]["f1-score"], 3), "support": int(rep[l]["support"])} for l in labels]
)
cm_df = pd.DataFrame(log["confusion_matrix"], columns=[f"pred {l}" for l in labels])
cm_df.insert(0, "true", labels)
t = log["test"]

EXAMPLES = [
    "ازيك يا صاحبي عامل ايه النهارده؟",
    "شلونك؟ وش تسوي اليوم؟",
    "كيفك؟ شو عم تعمل هلق؟",
    "شنوة حوالك؟ لاباس عليك؟",
    "أعلنت الحكومة اليوم عن خطة جديدة لتطوير التعليم.",
]

with gr.Blocks(title="Arabic Dialect ID") as demo:
    gr.Markdown(
        f"# 🗣️ Arabic Dialect ID\nAraBERT v0.2 fine-tuned to tell apart **Egyptian, Gulf, Levantine, "
        f"North African and MSA**. Test accuracy **{t['test_accuracy']:.3f}**, macro-F1 **{t['test_macro_f1']:.3f}**. "
        f"[Model card](https://huggingface.co/{MODEL_ID}) · [Trackio dashboard]({TRACKIO_URL})"
    )
    with gr.Tab("Try it"):
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(label="Arabic text", lines=3, rtl=True, placeholder="اكتب جملة هنا...")
                btn = gr.Button("Classify", variant="primary")
            out = gr.Label(label="Predicted dialect", num_top_classes=5)
        gr.Examples(EXAMPLES, inp)
        btn.click(predict, inp, out)
        inp.submit(predict, inp, out)
        gr.Markdown("⚠️ Short or mixed sentences are often genuinely ambiguous. See *Limitations* in the model card.")
    with gr.Tab("Training metrics"):
        gr.Markdown(f"Logged during training (also live on [Trackio]({TRACKIO_URL})).")
        with gr.Row():
            gr.LinePlot(train_df, x="step", y="loss", title="Training loss")
            gr.LinePlot(eval_df, x="step", y="value", color="metric", title="Dev metrics (per epoch)")
        gr.Markdown("### Test-set results per class")
        gr.Dataframe(class_df, interactive=False)
        gr.Markdown("### Confusion matrix (test)")
        gr.Dataframe(cm_df, interactive=False)

demo.launch()
