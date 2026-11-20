
"""Core functionality for the Legal Sentence Game."""

import re
from collections import defaultdict, Counter

# Sample legal sentences for training the N-gram model.
LEGAL_CORPUS = [
    "The parties agree to maintain confidentiality of all information.",
    "The tenant shall pay rent on or before the fifth day of each month.",
    "Either party may terminate this agreement by giving written notice.",
    "The company shall indemnify the employee against all legal claims.",
    "The parties shall resolve disputes through mutual arbitration.",
    "The agreement shall remain valid for a period of two years.",
    "The buyer shall make payment within thirty days of delivery.",
    "Neither party shall disclose confidential information to third parties.",
    "This agreement may be amended only by written consent.",
    "The supplier shall deliver the goods within the agreed period."
]

GAME_SENTENCES = [
    "The parties agree to maintain",
    "The tenant shall pay",
    "Either party may terminate",
    "The company shall indemnify",
    "The parties shall resolve"
]


# Milestone 1: Prepare legal text corpus.
def milestone_1(texts=None):
    """Clean and tokenize legal sentences."""
    if texts is None:
        texts = LEGAL_CORPUS

    processed = []

    for sentence in texts:
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
        words = sentence.split()

        if words:
            processed.append(words)

    return processed


# Milestone 2: Train an N-gram next-word predictor.
def milestone_2(corpus=None, n=2):
    """Train a simple N-gram model."""
    if corpus is None:
        corpus = milestone_1()

    model = defaultdict(Counter)

    for sentence in corpus:
        words = ["<START>"] * (n - 1) + sentence + ["<END>"]

        for i in range(n - 1, len(words)):
            context = tuple(words[i - n + 1:i])
            next_word = words[i]
            model[context][next_word] += 1

    return model


def predict_next(model, text, limit=5):
    """Predict possible next words from the given text."""
    words = text.lower().split()
    context = tuple(words[-1:])

    if context not in model:
        return []

    return [
        word for word, count in model[context].most_common(limit)
        if word != "<END>"
    ]


# Milestone 3: Language model classification.
def milestone_3(sentence):
    """Classify a legal sentence using fine-tuned GPT-2."""
    from app.classifier import classify
    return classify(sentence)

# Milestone 4: Prepare game questions.
def milestone_4():
    """Return incomplete legal sentences for the game."""
    return GAME_SENTENCES


# Milestone 5: Classify the answer and provide feedback.
def milestone_5(sentence):
    """Classify the completed sentence and return feedback."""
    try:
        category = milestone_3(sentence)
        feedback = (
            f"Your sentence has been classified as: {category}"
        )
        return {
            "category": category,
            "feedback": feedback
        }

    except Exception as error:
        return {
            "category": "Unknown",
            "feedback": f"Classification unavailable: {error}"
        }
