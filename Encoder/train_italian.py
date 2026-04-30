import json
import math
import os
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ACKNOWLEGEMENTS:
# We acknowlege the use of AI in writing this code. Specifically, we used AI
# to write the print statements, Result logging and Dataset loading sections, along with some 
# sections of the masked lanaguage modeling such as the Sinosoudal positional
# encoding. The language balancing feature was also inspired by AI suggestions. 

@dataclass
class TrainConfig:
    """Training configuration with linear probing options"""

    # ========== HARDWARE ==========
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== DATA FILES ==========
    french_pretrain_file: str = "PretrainingData/FrIt/Phonemes/fr_phonemes.jsonl"
    italian_pretrain_file: str = "PretrainingData/FrIt/Phonemes/it_phonemes.jsonl"
    intent_train_file: str = "FinetuningData/Italian/Phonemes/it_train_phonemes_400.jsonl"
    intent_test_file: str = "FinetuningData/Italian/Phonemes/it_test_phonemes_100.jsonl"

    # Where the per-epoch logs and final results go.
    # Each run writes to RESULTS/results_N.txt (auto-incremented).
    results_dir: str = "ITA_EPI_PRETRAIN_RESULTS"

    # ========== DATA SETTINGS ==========
    use_italian_data: bool = True
    french_data_limit: Optional[int] = 100000
    italian_data_limit: Optional[int] = None

    # ========== TRAINING HYPERPARAMETERS ==========
    batch_size: int = 32
    learning_rate: float = 5e-5
    pretrain_epochs: int = 30
    finetune_epochs: int = 30
    max_length: int = 256
    mlm_probability: float = 0.25

    # ========== MODEL ARCHITECTURE ==========
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # ========== EXPERIMENT CONTROL ==========
    use_pretraining: bool = True
    seed: int = 42

    # Early stopping
    use_early_stopping: bool = False
    patience: int = 15

    # ========== LANGUAGE BALANCING ==========
    balance_languages: bool = True
    italian_oversample_ratio: float = 0.3


# ============================================================
# 2. Logger: writes every line to console AND to results.txt
# ============================================================


class ResultsLogger:
    """
    Tee-style logger. Every call to log() prints to console AND appends to
    a per-run results file inside a directory.

    Each run gets its own file named results_N.txt where N is the next
    available integer. So your RESULTS/ folder ends up looking like:
        RESULTS/results_1.txt
        RESULTS/results_2.txt
        RESULTS/results_3.txt
        ...

    Usage:
        logger = ResultsLogger("RESULTS")
        logger.start_session(config)
        logger.log("Epoch 1 | Loss 1.23 | Acc 0.45")
        ...
        logger.end_session()
    """

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.path = self._next_results_path()
        self._fh = None

    def _next_results_path(self) -> Path:
        """Find the lowest unused results_N.txt in the results directory."""
        existing_numbers = []
        for p in self.results_dir.glob("results_*.txt"):
            stem = p.stem  # "results_3"
            try:
                n = int(stem.split("_", 1)[1])
                existing_numbers.append(n)
            except (ValueError, IndexError):
                # Skip files like "results_old.txt" or anything we can't parse.
                continue

        next_n = max(existing_numbers, default=0) + 1
        return self.results_dir / f"results_{next_n}.txt"

    def start_session(self, config: TrainConfig):
        # Open in write mode — this file is exclusively for this run.
        self._fh = open(self.path, "w", encoding="utf-8")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log(f"Run timestamp: {timestamp}")
        self.log(f"Output file:   {self.path}")
        self.log("")
        self.log("Config:")
        for k, v in asdict(config).items():
            self.log(f"  {k}: {v}")
        self.log("")

    def log(self, msg: str = ""):
        """Print to stdout and write to the results file."""
        print(msg)
        if self._fh is not None:
            self._fh.write(msg + "\n")
            self._fh.flush()  # so partial results survive a crash

    def end_session(self):
        if self._fh is not None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log("")
            self.log(f"Run finished: {timestamp}")
            self._fh.close()
            self._fh = None


# ============================================================
# 3. Tokenizer
# ============================================================


class PhonemeTokenizer:
    def __init__(self, phoneme_vocab: Optional[List[str]] = None):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"
        self.cls_token = "<cls>"
        self.sep_token = "<sep>"

        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.mask_token,
            self.cls_token,
            self.sep_token,
        ]

        if phoneme_vocab is None:
            phoneme_vocab = []

        vocab_items = self.special_tokens + [p for p in phoneme_vocab if p not in self.special_tokens]
        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(vocab_items)}
        self.id_to_token: Dict[int, str] = {i: tok for tok, i in self.token_to_id.items()}

        self.pad_id = self.token_to_id[self.pad_token]
        self.unk_id = self.token_to_id[self.unk_token]
        self.mask_id = self.token_to_id[self.mask_token]
        self.cls_id = self.token_to_id[self.cls_token]
        self.sep_id = self.token_to_id[self.sep_token]

    @classmethod
    def build_from_sequences(cls, sequences: List[str]) -> "PhonemeTokenizer":
        phoneme_set = set()
        for seq in sequences:
            for tok in seq.strip().split():
                if tok:
                    phoneme_set.add(tok)
        return cls(sorted(phoneme_set))

    def __len__(self) -> int:
        return len(self.token_to_id)

    def tokenize(self, sequence: str) -> List[str]:
        return [tok for tok in sequence.strip().split() if tok]

    def encode(self, sequence: str, max_length: Optional[int] = None, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(sequence)
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]

        if add_special_tokens:
            ids = [self.cls_id] + ids + [self.sep_id]

        if max_length is not None and len(ids) > max_length:
            if add_special_tokens:
                ids = ids[:max_length - 1] + [self.sep_id]
            else:
                ids = ids[:max_length]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for idx in ids:
            tok = self.id_to_token.get(int(idx), self.unk_token)
            if skip_special_tokens and tok in self.special_tokens:
                continue
            tokens.append(tok)
        return " ".join(tokens)


# ============================================================
# 4. Dataset loading helpers
# ============================================================


def load_jsonl(path: str) -> List[dict]:
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
                    print(f"Warning: Skipping malformed line {line_num} in {path}: {e}")
    except FileNotFoundError:
        print(f"Error: File {path} not found!")
        return []
    return records


class LabelEncoder:
    def __init__(self, labels: List[str]):
        uniq = sorted(set(labels))
        self.label_to_id = {label: i for i, label in enumerate(uniq)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}

    def encode(self, label: str) -> int:
        return self.label_to_id[label]

    def decode(self, idx: int) -> str:
        return self.id_to_label[idx]

    def __len__(self) -> int:
        return len(self.label_to_id)


# ============================================================
# 5. Dataset classes
# ============================================================


class MLMDataset(Dataset):
    def __init__(self, phoneme_sequences: List[str], tokenizer: PhonemeTokenizer, max_length: int = 256):
        self.phoneme_sequences = phoneme_sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.phoneme_sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.phoneme_sequences[idx]
        input_ids = self.tokenizer.encode(seq, max_length=self.max_length, add_special_tokens=True)
        return {"input_ids": input_ids}


class IntentDataset(Dataset):
    def __init__(
        self,
        records: List[dict],
        tokenizer: PhonemeTokenizer,
        label_encoder: LabelEncoder,
        max_length: int = 256,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        item = self.records[idx]
        input_ids = self.tokenizer.encode(item["utt"], max_length=self.max_length, add_special_tokens=True)
        label_id = self.label_encoder.encode(item["intent"])
        return {
            "input_ids": input_ids,
            "label": label_id,
            "intent": item["intent"],
            "utt": item.get("utt", ""),
        }


# ============================================================
# 6. Collators
# ============================================================


def pad_sequences(sequences: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, seq in enumerate(sequences):
        length = len(seq)
        input_ids[i, :length] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :length] = 1

    return input_ids, attention_mask


class MLMCollator:
    def __init__(self, tokenizer: PhonemeTokenizer, mlm_probability: float = 0.25):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, batch: List[dict]) -> dict:
        sequences = [item["input_ids"] for item in batch]
        input_ids, attention_mask = pad_sequences(sequences, self.tokenizer.pad_id)

        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_token_mask = (
            (input_ids == self.tokenizer.pad_id)
            | (input_ids == self.tokenizer.cls_id)
            | (input_ids == self.tokenizer.sep_id)
        )
        probability_matrix.masked_fill_(special_token_mask, 0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        rand = torch.rand(labels.shape)

        replace_mask = masked_indices & (rand < 0.8)
        input_ids[replace_mask] = self.tokenizer.mask_id

        random_mask = masked_indices & (rand >= 0.8) & (rand < 0.9)
        random_tokens = torch.randint(low=0, high=len(self.tokenizer), size=labels.shape)
        input_ids[random_mask] = random_tokens[random_mask]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class IntentCollator:
    def __init__(self, tokenizer: PhonemeTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[dict]) -> dict:
        sequences = [item["input_ids"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        input_ids, attention_mask = pad_sequences(sequences, self.tokenizer.pad_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "intents": [item["intent"] for item in batch],
            "utterances": [item["utt"] for item in batch],
        }


# ============================================================
# 7. Positional encoding
# ============================================================


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# ============================================================
# 8. Model architecture
# ============================================================


class PhonemeEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 1024,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.position(x)
        x = self.dropout(x)

        src_key_padding_mask = attention_mask == 0
        hidden_states = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return hidden_states


class PhonemeMLMModel(nn.Module):
    def __init__(self, encoder: PhonemeEncoder, vocab_size: int):
        super().__init__()
        self.encoder = encoder
        self.lm_head = nn.Linear(encoder.d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.encoder(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        return logits


class PhonemeIntentClassifier(nn.Module):
    def __init__(self, encoder: PhonemeEncoder, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.d_model, num_labels)

    def masked_mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-8)
        return summed / denom

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.encoder(input_ids, attention_mask)
        pooled = self.masked_mean_pool(hidden_states, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# ============================================================
# 9. Training utilities
# ============================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_balanced_sampler(french_sequences: List[str], italian_sequences: List[str],
                            italian_ratio: float = 0.3) -> Tuple[List[str], List[float]]:
    combined_sequences = french_sequences + italian_sequences
    language_labels = [0] * len(french_sequences) + [1] * len(italian_sequences)

    n_french = len(french_sequences)
    n_italian = len(italian_sequences)
    n_total = n_french + n_italian

    french_weight = (1 - italian_ratio) / (n_french / n_total)
    italian_weight = italian_ratio / (n_italian / n_total)

    weights = []
    for lang_id in language_labels:
        if lang_id == 0:
            weights.append(french_weight)
        else:
            weights.append(italian_weight)

    print(f"Language balancing:")
    print(f"  French sequences: {n_french:,}")
    print(f"  Italian sequences: {n_italian:,}")
    print(f"  Target Italian ratio: {italian_ratio:.1%}")
    print(f"  French weight: {french_weight:.3f}")
    print(f"  Italian weight: {italian_weight:.3f}")

    return combined_sequences, weights


def train_mlm_epoch(model: nn.Module, loader: DataLoader, optimizer, device: str) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def train_classifier_epoch(model: nn.Module, loader: DataLoader, optimizer, device: str) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_classifier(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, dict]:
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

    accuracy = correct / max(total, 1)

    metrics = {
        "accuracy": accuracy,
        "total_examples": total,
        "predictions": predictions,
        "true_labels": true_labels,
    }

    return accuracy, metrics


# ============================================================
# 10. Main training pipeline
# ============================================================


def main():
    config = TrainConfig()
    set_seed(config.seed)

    logger = ResultsLogger(config.results_dir)
    logger.start_session(config)

    logger.log("Phoneme-based Intent Classification with Linear Probing")
    logger.log("=" * 60)
    logger.log(f"Device: {config.device}")
    logger.log(f"Use pretraining: {config.use_pretraining}")
    logger.log(f"Seed: {config.seed}")
    logger.log("")

    # --------------------------------------------------------
    # STEP 1: Load data
    # --------------------------------------------------------
    logger.log("Loading data files...")

    french_records = load_jsonl(config.french_pretrain_file)
    if not french_records:
        logger.log("Error: Could not load French phoneme data!")
        logger.end_session()
        return

    italian_records = []
    if config.use_italian_data:
        italian_records = load_jsonl(config.italian_pretrain_file)
        if not italian_records:
            logger.log("Warning: Could not load Italian data. Continuing with French only.")
            config.use_italian_data = False

    finetune_train_records = load_jsonl(config.intent_train_file)
    finetune_test_records = load_jsonl(config.intent_test_file)

    if not finetune_train_records or not finetune_test_records:
        logger.log("Error: Could not load intent classification data!")
        logger.end_session()
        return

    logger.log(f"Loaded {len(french_records):,} French sequences")
    if config.use_italian_data:
        logger.log(f"Loaded {len(italian_records):,} Italian sequences")
    logger.log(f"Loaded {len(finetune_train_records):,} training examples")
    logger.log(f"Loaded {len(finetune_test_records):,} test examples")

    if config.french_data_limit and len(french_records) > config.french_data_limit:
        french_records = french_records[:config.french_data_limit]
        logger.log(f"Limited French data to {len(french_records):,} sequences")

    if config.italian_data_limit and len(italian_records) > config.italian_data_limit:
        italian_records = italian_records[:config.italian_data_limit]
        logger.log(f"Limited Italian data to {len(italian_records):,} sequences")

    french_sequences = [item["text"] for item in french_records]
    italian_sequences = [item["text"] for item in italian_records] if config.use_italian_data else []
    train_sequences = [item["utt"] for item in finetune_train_records]
    test_sequences = [item["utt"] for item in finetune_test_records]

    # --------------------------------------------------------
    # STEP 2: Build tokenizer and label encoder
    # --------------------------------------------------------
    logger.log("\nBuilding tokenizer...")
    all_sequences = french_sequences + italian_sequences + train_sequences + test_sequences
    tokenizer = PhonemeTokenizer.build_from_sequences(all_sequences)
    logger.log(f"Vocabulary size: {len(tokenizer)}")

    label_encoder = LabelEncoder([item["intent"] for item in finetune_train_records])
    logger.log(f"Intent classes: {len(label_encoder)}")
    logger.log(f"Intents: {list(label_encoder.label_to_id.keys())}")
    logger.log("")

    if config.use_pretraining:
        # --------------------------------------------------------
        # STEP 3: Pretraining
        # --------------------------------------------------------
        logger.log("Setting up pretraining...")

        if config.use_italian_data and config.balance_languages:
            combined_sequences, sampling_weights = create_balanced_sampler(
                french_sequences, italian_sequences, config.italian_oversample_ratio
            )
            mlm_dataset = MLMDataset(combined_sequences, tokenizer, max_length=config.max_length)
            sampler = WeightedRandomSampler(
                weights=sampling_weights,
                num_samples=len(sampling_weights),
                replacement=True,
            )
            mlm_loader = DataLoader(
                mlm_dataset,
                batch_size=config.batch_size,
                sampler=sampler,
                collate_fn=MLMCollator(tokenizer, mlm_probability=config.mlm_probability),
            )
        else:
            all_pretrain_sequences = french_sequences + italian_sequences
            mlm_dataset = MLMDataset(all_pretrain_sequences, tokenizer, max_length=config.max_length)
            mlm_loader = DataLoader(
                mlm_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=MLMCollator(tokenizer, mlm_probability=config.mlm_probability),
            )

        encoder = PhonemeEncoder(
            vocab_size=len(tokenizer),
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_len=config.max_length * 2,
            pad_id=tokenizer.pad_id,
        )
        mlm_model = PhonemeMLMModel(encoder=encoder, vocab_size=len(tokenizer)).to(config.device)
        mlm_optimizer = torch.optim.AdamW(mlm_model.parameters(), lr=config.learning_rate)

        logger.log(f"Model parameters: {sum(p.numel() for p in mlm_model.parameters()):,}")
        logger.log(f"Training batches per epoch: {len(mlm_loader):,}")
        logger.log("")

        logger.log("Starting MLM pretraining...")
        best_loss = float('inf')

        for epoch in range(config.pretrain_epochs):
            loss = train_mlm_epoch(mlm_model, mlm_loader, mlm_optimizer, config.device)
            logger.log(f"[Pretrain] Epoch {epoch + 1:2d}/{config.pretrain_epochs} | MLM Loss: {loss:.4f}")
            if loss < best_loss:
                best_loss = loss

        logger.log(f"Pretraining completed. Best MLM loss: {best_loss:.4f}")
        logger.log("")

    else:
        logger.log("Creating fresh encoder (no pretraining)...")
        encoder = PhonemeEncoder(
            vocab_size=len(tokenizer),
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            max_len=config.max_length * 2,
            pad_id=tokenizer.pad_id,
        )
        mlm_model = PhonemeMLMModel(encoder=encoder, vocab_size=len(tokenizer)).to(config.device)

    # --------------------------------------------------------
    # STEP 4: Fine-tuning setup
    # --------------------------------------------------------
    logger.log("Setting up fine-tuning...")

    train_dataset = IntentDataset(
        records=finetune_train_records,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_length=config.max_length,
    )
    test_dataset = IntentDataset(
        records=finetune_test_records,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        max_length=config.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=IntentCollator(tokenizer),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=IntentCollator(tokenizer),
    )

    classifier = PhonemeIntentClassifier(
        encoder=mlm_model.encoder,
        num_labels=len(label_encoder),
        dropout=config.dropout,
    ).to(config.device)

    logger.log(f"Training examples: {len(train_dataset):,}")
    logger.log(f"Test examples: {len(test_dataset):,}")
    logger.log("")

    # --------------------------------------------------------
    # STEP 5: Fine-tuning
    # --------------------------------------------------------
    best_accuracy = 0.0
    epochs_without_improvement = 0
    test_accuracies: List[float] = []

    logger.log("=" * 60)
    logger.log("Full Fine-tuning (All Parameters)")
    logger.log("=" * 60)

    clf_optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )

    for epoch in range(config.finetune_epochs):
        train_loss = train_classifier_epoch(classifier, train_loader, clf_optimizer, config.device)
        test_acc, _ = evaluate_classifier(classifier, test_loader, config.device)
        test_accuracies.append(test_acc)

        logger.log(f"[Finetune] Epoch {epoch + 1:3d}/{config.finetune_epochs} | "
                   f"Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if config.use_early_stopping and epochs_without_improvement >= config.patience:
            logger.log(f"Early stopping at epoch {epoch + 1}")
            break

    # --------------------------------------------------------
    # STEP 6: Final evaluation
    # --------------------------------------------------------
    final_accuracy, final_metrics = evaluate_classifier(classifier, test_loader, config.device)

    logger.log("")
    logger.log("=" * 60)
    logger.log("FINAL RESULTS")
    logger.log("=" * 60)
    logger.log(f"Best Test Accuracy: {best_accuracy:.4f}")
    logger.log(f"Final Test Accuracy: {final_accuracy:.4f}")
    if test_accuracies:
        last_n = test_accuracies[-5:]
        avg_last_5 = sum(last_n) / len(last_n)
        logger.log(f"Avg Test Accuracy (last {len(last_n)} epochs): {avg_last_5:.4f}")
    logger.log(f"Total Test Examples: {final_metrics['total_examples']:,}")
    logger.log(f"Pretraining Used: {config.use_pretraining}")
    if config.use_italian_data:
        logger.log(f"French Sequences: {len(french_sequences):,}")
        logger.log(f"Italian Sequences: {len(italian_sequences):,}")
    logger.log("Training completed!")

    logger.end_session()


if __name__ == "__main__":
    main()