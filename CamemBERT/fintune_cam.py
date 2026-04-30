"""
This script fine-tunes CamemBERT for intent classification on the Haitian/Italian dataset.
Does not deal with phonemes, but just raw italian or haitian text. 
"""

import json
import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    CamembertTokenizer,
    CamembertModel,
    CamembertForSequenceClassification,
    get_linear_schedule_with_warmup
)

# ACKNOWLEGEMENTS:
# We acknowlege the use of AI in writing this code. Specifically, we used AI
# to write the Result logging, print statements, and Dataset loading sections
# along with some sections of how to get the CamemBERT weights 
# The language balancing feature was also inspired by AI suggestions. 

@dataclass
class CamemBERTConfig:
    """Configuration for CamemBERT fine-tuning"""
    
    # ========== MODEL ==========
    model_name: str = "camembert-base" 
    
    # ========== DATA ==========
    train_file: str = "FinetuningData/Haitian/ht_train_400.jsonl"  
    test_file: str = "FinetuningData/Haitian/ht_test_100.jsonl"    
    max_length: int = 128 
    
    # ========== TRAINING ==========
    batch_size: int = 16  
    learning_rate: float = 2e-5 
    num_epochs: int = 50
    warmup_steps: int = 100  
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  
    
    # ========== EXPERIMENT ==========
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_best_model: bool = True
    output_dir: str = "camembert_haitian_intent"
    
    # ========== EARLY STOPPING ==========
    use_early_stopping: bool = False
    patience: int = 3  


# ============================================================
# 2. Data Loading
# ============================================================


def load_jsonl(path: str) -> List[dict]:
    """Load JSONL file"""
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line {line_num}: {e}")
    except FileNotFoundError:
        print(f"Error: File {path} not found!")
        return []
    return records


class IntentDataset(Dataset):
    """Dataset for intent classification with CamemBERT tokenization"""
    
    def __init__(
        self,
        records: List[dict],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.records[idx]
        
        # Get text and label
        # IMPORTANT: Using "utt" field which contains the actual Haitian text
        # NOT phonemes - this is raw text that CamemBERT will tokenize
        text = item["utt"]
        label = item["intent"]
        
        # Tokenize using CamemBERT tokenizer
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # Remove batch dimension
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.label2id[label], dtype=torch.long),
            "intent": label,
            "text": text,
        }


# ============================================================
# 3. Training & Evaluation
# ============================================================


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    max_grad_norm: float = 1.0
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: str) -> Tuple[float, Dict]:
    """Evaluate model"""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        preds = torch.argmax(logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        total_loss += loss.item()
        num_batches += 1
    
    # Calculate accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels)
    avg_loss = total_loss / num_batches
    
    return accuracy, {
        "accuracy": accuracy,
        "loss": avg_loss,
        "predictions": all_preds,
        "labels": all_labels,
    }


# ============================================================
# 4. Main Training Loop
# ============================================================


def main():
    config = CamemBERTConfig()
    set_seed(config.seed)
    
    print("=" * 70)
    print("CamemBERT Fine-tuning on Haitian Intent Classification")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Seed: {config.seed}")
    print()
    
    # --------------------------------------------------------
    # Load Data
    # --------------------------------------------------------
    print("Loading data...")
    train_records = load_jsonl(config.train_file)
    test_records = load_jsonl(config.test_file)
    
    if not train_records or not test_records:
        print("Error: Could not load data files!")
        print(f"Make sure {config.train_file} and {config.test_file} exist")
        print("Expected format: {'utt': 'text here', 'intent': 'intent_class'}")
        return
    
    print(f"Loaded {len(train_records)} training examples")
    print(f"Loaded {len(test_records)} test examples")
    
    # Build label mapping
    all_intents = list(set(item["intent"] for item in train_records))
    label2id = {label: idx for idx, label in enumerate(sorted(all_intents))}
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(label2id)
    
    print(f"Number of intent classes: {num_labels}")
    print(f"Intent classes: {list(label2id.keys())}")
    print()
    
    # --------------------------------------------------------
    # Load Tokenizer and Model
    # --------------------------------------------------------
    print(f"Loading CamemBERT tokenizer and model...")
    
    try:
        tokenizer = CamembertTokenizer.from_pretrained(config.model_name)
    except:
        # Fallback to HuggingFace model name
        print(f"Trying alternative model name: camembert/camembert-base")
        config.model_name = "camembert/camembert-base"
        tokenizer = CamembertTokenizer.from_pretrained(config.model_name)
    
    # Load model with classification head
    model = CamembertForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model.to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # --------------------------------------------------------
    # Create Datasets and Dataloaders
    # --------------------------------------------------------
    print("Creating datasets...")
    train_dataset = IntentDataset(train_records, tokenizer, label2id, config.max_length)
    test_dataset = IntentDataset(test_records, tokenizer, label2id, config.max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    
    print(f"Training batches per epoch: {len(train_loader)}")
    print()
    
    # --------------------------------------------------------
    # Setup Optimizer and Scheduler
    # --------------------------------------------------------
    # AdamW optimizer (standard for BERT fine-tuning)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler with linear warmup
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Total training steps: {total_steps:,}")
    print(f"Warmup steps: {config.warmup_steps}")
    print()
    
    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    print("Starting training...")
    print("=" * 70)
    
    best_accuracy = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(config.num_epochs):
        # Train
        train_loss = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            scheduler, 
            config.device,
            config.max_grad_norm
        )
        
        # Evaluate
        test_acc, test_metrics = evaluate(model, test_loader, config.device)
        
        print(f"Epoch {epoch + 1:2d}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Acc: {test_acc:.4f} | "
              f"Test Loss: {test_metrics['loss']:.4f}")
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            epochs_without_improvement = 0
            
            if config.save_best_model:
                save_path = f"{config.output_dir}_best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': test_acc,
                    'label2id': label2id,
                    'config': config,
                }, save_path)
                print(f"  → Saved best model (accuracy: {test_acc:.4f})")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if config.use_early_stopping and epochs_without_improvement >= config.patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {config.patience} epochs)")
            break
    
    # --------------------------------------------------------
    # Final Evaluation
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best Test Accuracy: {best_accuracy:.4f}")
    print(f"Model: {config.model_name}")
    print(f"Training examples: {len(train_records)}")
    print(f"Test examples: {len(test_records)}")
    print(f"Number of intent classes: {num_labels}")
    print()
    
    # Save final model
    final_save_path = f"{config.output_dir}_final_seed{config.seed}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'accuracy': best_accuracy,
        'label2id': label2id,
        'id2label': id2label,
        'config': config,
    }, final_save_path)
    print(f"Saved final model to: {final_save_path}")
    
    return best_accuracy


# ============================================================
# 5. Inference Function (Optional)
# ============================================================


def load_model_for_inference(model_path: str, device: str = "cuda"):
    """Load a saved model for inference"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    label2id = checkpoint['label2id']
    id2label = checkpoint['id2label']
    
    # Load tokenizer and model
    tokenizer = CamembertTokenizer.from_pretrained(config.model_name)
    model = CamembertForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer, id2label


@torch.no_grad()
def predict_intent(text: str, model, tokenizer, id2label, device: str = "cuda") -> Tuple[str, float]:
    """Predict intent for a single text"""
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    probs = torch.softmax(logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, pred_idx].item()
    
    predicted_intent = id2label[pred_idx]
    
    return predicted_intent, confidence


# ============================================================
# 6. Main Entry Point
# ============================================================


if __name__ == "__main__":
    # Train the model
    best_acc = main()
    
    print()
    print("=" * 70)
    print("Training complete!")
    print(f"Best accuracy: {best_acc:.4f}")
    print("=" * 70)
    
    # Example inference (optional)
    print("\nExample inference:")
    print("To use the trained model for prediction:")
    print("""
    model, tokenizer, id2label = load_model_for_inference('camembert_haitian_intent_best.pt')
    intent, confidence = predict_intent('jwe playlist pre-jwèt la', model, tokenizer, id2label)
    print(f'Predicted intent: {intent} (confidence: {confidence:.2f})')
    """)