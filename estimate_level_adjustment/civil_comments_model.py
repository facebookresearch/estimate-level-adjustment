# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ============================================================
# 1. DATASET CLASS
# ============================================================
class ToxicityDataset(Dataset):
    texts: NDArray[Any]
    labels: NDArray[Any]
    tokenizer: Any
    max_length: int

    def __init__(
        self,
        texts: NDArray[Any],
        labels: NDArray[Any],
        tokenizer: Any,
        max_length: int = 128,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float),
        }


# ============================================================
# 2. MODEL CLASS
# ============================================================
class ToxicityClassifier(nn.Module):
    base_model: Any
    classifier: nn.Linear

    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int = 1,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits.squeeze(-1)


# ============================================================
# 3. TRAINING FUNCTION
# ============================================================
def train_epoch(
    model: ToxicityClassifier,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================
# 4. EVALUATION FUNCTION
# ============================================================
def evaluate(
    model: ToxicityClassifier,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Convert logits to predictions (sigmoid for binary)
            preds = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    # Calculate accuracy (threshold at 0.5)
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    accuracy = ((predictions > threshold) == true_labels).mean()

    return total_loss / len(dataloader), accuracy


# ============================================================
# 5. MAIN TRAINING SCRIPT
# ============================================================


def train_toxicity_model(
    train_df: pd.DataFrame,
    model_path: str = "bert-base-uncased",
    text_column: str = "comment_text",
    label_column: str = "target",
    batch_size: int = 16,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    val_size: float = 0.2,
    patience_epochs: int = 3,
    save_path: str = "toxicity_model.pt",
    local_files_only: bool = False,
) -> tuple[ToxicityClassifier, Any, dict[str, list[float]]]:
    """
    Train a toxicity classifier on the given dataframe.

    train_df: DataFrame with text and label columns
    model_path: Path or name of HuggingFace model (default: "bert-base-uncased")
    text_column: Name of the text column in train_df
    label_column: Name of the label column in train_df
    :param batch_size: Training batch size
    :param epochs: Maximum number of training epochs
    :param learning_rate: Learning rate for AdamW optimizer
    :param max_length: Maximum sequence length for tokenization
    :param val_size: Fraction of data to use for validation
    :param patience_epochs: Early stopping patience
    :param save_path: Path to save the trained model
    :param local_files_only: Whether to load model from local files only
    :return: Trained model and training history
    """

    # --- Setup device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=local_files_only
    )

    # --- Prepare data ---
    texts = train_df[text_column].values
    labels = train_df[label_column].values

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=val_size, random_state=42
    )

    print(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")

    # --- Create datasets and dataloaders ---
    train_dataset = ToxicityDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = ToxicityDataset(val_texts, val_labels, tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # --- Initialize model ---
    model = ToxicityClassifier(
        model_path, num_classes=1, local_files_only=local_files_only
    )
    model.to(device)

    # --- Loss and optimizer ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, foreach=True)

    # --- Training state ---
    best_val_loss = float("inf")
    best_model_state = None
    patience = patience_epochs
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    # --- Training loop ---
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience = patience_epochs
            print(">>> New Best Model!")
        else:
            patience -= 1
            print(f"Early Stopping Patience Left: {patience}")
            if patience == 0:
                print(">>> Early Stopping Triggered!")
                break

    # --- Load best model and save ---
    if best_model_state:
        model.load_state_dict(best_model_state)

    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    return model, tokenizer, history


def predict_toxicity_text(
    model: ToxicityClassifier,
    tokenizer: Any,
    text: str,
    device: Optional[torch.device] = None,
    max_length: int = 128,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probability = torch.sigmoid(output).item()

    return {
        "text": text,
        "toxic_probability": probability,
        "is_toxic": probability > threshold,
    }


def load_toxicity_model(
    model_path: str = "bert-base-uncased",
    weights_path: str = "toxicity_model.pt",
    device: Optional[torch.device] = None,
    local_files_only: bool = False,
) -> tuple[ToxicityClassifier, Any, torch.device]:
    """
    Load a trained toxicity classifier.

    :param model_path: Path or name of HuggingFace model (for architecture)
    :param weights_path: Path to saved model weights (.pt file)
    :param device: torch device (auto-detected if None)
    :param local_files_only: Whether to load base model from local files only
    :return: model, tokenizer, device
    """
    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=local_files_only
    )

    # Recreate model architecture
    model = ToxicityClassifier(
        model_path, num_classes=1, local_files_only=local_files_only
    )

    # Load trained weights
    model.load_state_dict(
        torch.load(weights_path, weights_only=True, map_location=device)
    )
    model.to(device)
    model.eval()

    print(f"Model loaded from {weights_path}")
    print(f"Using device: {device}")

    return model, tokenizer, device


def predict_toxicity_texts(
    model: ToxicityClassifier,
    tokenizer: Any,
    texts: list[str],
    device: Optional[torch.device] = None,
    max_length: int = 128,
    batch_size: int = 32,
) -> NDArray[np.floating[Any]]:
    """
    Predict toxicity scores for a list of texts.

    :param model: Trained ToxicityClassifier
    :param tokenizer: Tokenizer
    :param texts: List of strings to classify
    :param device: torch device
    :param max_length: Max sequence length
    :param batch_size: Batch size for inference
    :return: Array of toxicity probabilities
    """
    model.eval()
    all_preds = []
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        encoding = tokenizer(
            list(batch_texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)

    return np.array(all_preds)


def plot_training_history(history: dict[str, list[float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(history["val_accuracy"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")

    plt.tight_layout()
    plt.show()
