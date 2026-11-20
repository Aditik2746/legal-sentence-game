
"""Load and use the fine-tuned GPT-2 legal clause classifier."""

import torch
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "legal-gpt2"

# Load the model only once.
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2ForSequenceClassification.from_pretrained(MODEL_PATH)

model.config.pad_token_id = tokenizer.pad_token_id
model.eval()


def classify(sentence):
    """Predict the legal clause category of a sentence."""

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_id = torch.argmax(outputs.logits, dim=1).item()

    return model.config.id2label[predicted_id]