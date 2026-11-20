# Legal Sentence Game

A command-line application that helps law students practice completing legal sentences. It uses an N-gram language model to suggest the next word and a fine-tuned GPT-2 model to classify completed sentences into legal clause categories.

## Features

- **Legal Text Preprocessing:** Cleans and tokenizes legal sentences.
- **Next-Word Prediction:** Uses a bigram N-gram model to suggest possible next words.
- **GPT-2 Classification:** Uses a fine-tuned GPT-2 model to classify completed legal sentences.
- **Interactive CLI:** Allows users to complete five partial legal sentences.
- **Instant Feedback:** Displays the predicted clause category after each answer.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- GPT-2
- N-gram Language Model
- Pytest

## Legal Clause Categories

The classifier predicts one of these eight categories:

1. Confidentiality
2. Payment
3. Termination
4. Indemnity
5. Dispute Resolution
6. Duration
7. Delivery
8. Other

## Project Structure

```text
legal-sentence-game/
│
├── app/
│   ├── __init__.py
│   ├── cli.py
│   ├── core.py
│   ├── classifier.py
│   └── storage.py
│
├── tests/
│   ├── test_core.py
│   └── test_smoke.py
│
├── models/
│   └── legal-gpt2/
│
├── train_model.py
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd legal-sentence-game
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Model Training

The project uses pretrained GPT-2 for legal clause classification.

Run the following command to fine-tune GPT-2 using the included legal sentence dataset:

```bash
python train_model.py
```

The trained model and tokenizer are saved in:

```text
models/legal-gpt2/
```

**Note:** The model directory is excluded from Git because of its large size. Run the training script before starting the game if the saved model is not present.

## Run the Game

```bash
python -m app.cli play
```

The application presents five incomplete legal sentences. Users complete them, and the GPT-2 classifier predicts the clause category.

To check the project setup:

```bash
python -m app.cli hello
```

## Run Tests

Execute the automated test suite:

```bash
python -m pytest
```

## How It Works

1. Legal sentences are cleaned and tokenized during preprocessing.
2. An N-gram model learns word sequences and suggests possible next words.
3. The user completes a partial legal sentence.
4. The completed sentence is passed to the fine-tuned GPT-2 classifier.
5. The classifier predicts a legal clause category.
6. The application displays feedback.

## Limitations

The current classifier is fine-tuned on a small illustrative dataset. Its predictions are intended for educational demonstration and may not generalize reliably to real-world legal documents.

This project is not intended to provide legal advice.

## Future Improvements

- Train using a larger real-world legal text dataset.
- Improve classification accuracy.
- Add more legal clause categories.
- Develop a graphical user interface.
- Evaluate the model using accuracy, precision, recall and F1-score.

## Author

Developed as an academic project to explore Natural Language Processing, language models and legal text classification.