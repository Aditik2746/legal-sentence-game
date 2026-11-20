
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from pathlib import Path

# 1. Define legal clause categories
labels = [
    "Confidentiality",
    "Payment",
    "Termination",
    "Indemnity",
    "Dispute Resolution",
    "Duration",
    "Delivery",
    "Other"
]

label_to_id = {label: i for i, label in enumerate(labels)}

# 2. Small training dataset
# Each sentence is paired with its legal clause category.
data = [
    ("The parties shall maintain confidentiality of all information.", "Confidentiality"),
    ("Neither party shall disclose confidential information.", "Confidentiality"),
    ("All business information must remain private.", "Confidentiality"),
    ("The employee agrees not to share company secrets.", "Confidentiality"),
    ("The parties shall protect all confidential documents.", "Confidentiality"),

    ("The tenant shall pay rent every month.", "Payment"),
    ("Payment must be completed within thirty days.", "Payment"),
    ("The buyer shall pay the agreed amount.", "Payment"),
    ("The customer must clear all outstanding dues.", "Payment"),
    ("The invoice shall be paid before the due date.", "Payment"),

    ("Either party may terminate this agreement.", "Termination"),
    ("This contract can be terminated by written notice.", "Termination"),
    ("The company may end the agreement at any time.", "Termination"),
    ("Either party may cancel the contract with notice.", "Termination"),
    ("The agreement shall terminate upon expiration.", "Termination"),

    ("The company shall indemnify the employee against claims.", "Indemnity"),
    ("The supplier shall compensate the buyer for losses.", "Indemnity"),
    ("The contractor shall protect the company from legal claims.", "Indemnity"),
    ("The party shall be responsible for damages caused.", "Indemnity"),
    ("The company agrees to cover all legal expenses.", "Indemnity"),

    ("Disputes shall be resolved through arbitration.", "Dispute Resolution"),
    ("The parties agree to settle disputes amicably.", "Dispute Resolution"),
    ("Any disagreement shall be referred to mediation.", "Dispute Resolution"),
    ("The courts shall resolve disputes under applicable law.", "Dispute Resolution"),
    ("Both parties shall attempt to resolve conflicts peacefully.", "Dispute Resolution"),

    ("This agreement shall remain valid for two years.", "Duration"),
    ("The contract shall continue for a period of twelve months.", "Duration"),
    ("The agreement will remain effective until December.", "Duration"),
    ("This contract shall be valid for five years.", "Duration"),
    ("The term of this agreement is three years.", "Duration"),

    ("The supplier shall deliver the goods within thirty days.", "Delivery"),
    ("Products must be delivered to the buyer.", "Delivery"),
    ("The company shall deliver the materials on time.", "Delivery"),
    ("The goods shall be shipped to the agreed address.", "Delivery"),
    ("The seller must deliver all items in good condition.", "Delivery"),

    ("The sky is blue today.", "Other"),
    ("Students are preparing for their examinations.", "Other"),
    ("The meeting will begin tomorrow morning.", "Other"),
    ("The weather is pleasant this afternoon.", "Other"),
    ("The office is located in Bengaluru.", "Other")
]


# 3. Prepare the dataset
class LegalDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, label = self.data[index]

        encoding = self.tokenizer(
            sentence,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_to_id[label], dtype=torch.long)
        }


def main():
    torch.set_num_threads(2)

    print("Loading GPT-2 tokenizer...")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading GPT-2 classification model...")

    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2",
        num_labels=len(labels),
        id2label={i: label for i, label in enumerate(labels)},
        label2id=label_to_id
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    dataset = LegalDataset(data, tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5
    )

    # 4. Fine-tune GPT-2
    model.train()
    epochs = 2

    for epoch in range(epochs):
        total_loss = 0

        for batch in loader:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f}"
        )

    # 5. Save the fine-tuned model
    output_dir = Path("models/legal-gpt2")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nGPT-2 fine-tuning completed!")
    print(f"Model saved at: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
