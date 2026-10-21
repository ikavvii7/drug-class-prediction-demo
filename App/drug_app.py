from pathlib import Path
import pandas as pd
import gradio as gr
import skops.io as sio
import os

ROOT = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    ROOT / "Model" / "drug_pipeline.skops",       # Hugging Face Space layout
    ROOT / "drug_pipeline.skops",                  # Alternative Space root layout
    ROOT.parent / "Model" / "drug_pipeline.skops", # Local repository layout
]
MODEL_PATH = next((path for path in MODEL_CANDIDATES if path.exists()), None)

if MODEL_PATH is None:
    expected = "\n".join(str(path) for path in MODEL_CANDIDATES)
    raise FileNotFoundError(
        f"Model file not found. Checked:\n{expected}\nRun `python train.py` first."
    )

# Only load artifacts produced by this project's trusted training workflow.
# Review unknown types before loading any model received from an untrusted source.
unknown_types = sio.get_untrusted_types(file=MODEL_PATH)
pipe = sio.load(MODEL_PATH, trusted=unknown_types)

FEATURES = list(pipe.feature_names_in_)
NUMERIC = set()
CATEGORICAL_CHOICES = {
    "Sex": ["F", "M"],
    "BP": ["LOW", "NORMAL", "HIGH"],
    "Cholesterol": ["NORMAL", "HIGH"],
}

def predict(*values):
    row = {}
    for name, value in zip(FEATURES, values):
        row[name] = value
    frame = pd.DataFrame([row], columns=FEATURES)
    result = pipe.predict(frame)[0]
    return f"Predicted class: **{result}**\n\nEducational demo only—not medical advice."

components = []
for name in FEATURES:
    if name in CATEGORICAL_CHOICES:
        components.append(gr.Radio(
            choices=CATEGORICAL_CHOICES[name],
            value=CATEGORICAL_CHOICES[name][0],
            label=name
        ))
    else:
        if name == "Age":
            components.append(gr.Slider(15, 100, value=40, step=1, label=name))
        elif name == "Na_to_K":
            components.append(gr.Slider(0.1, 50.0, value=15.0, step=0.1, label=name))
        else:
            components.append(gr.Number(value=0, label=name))

demo = gr.Interface(
    fn=predict,
    inputs=components,
    outputs=gr.Markdown(),
    title="Drug Class Prediction Demo",
    description=(
        "Enter the dataset features to see the model's predicted class. "
        "This is an educational machine-learning example and must not be used "
        "to choose or change medication."
    ),
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )