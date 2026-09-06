"""Fine-tuning training runner for Sanskrit OCR using PaddlePaddle & CTC Loss.

Trains a CRNN / sequence recognition model on multi-domain Sanskrit text lines
(digital, handwritten, and ancient manuscripts) and exports fine-tuned checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure safe console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Character Dictionary & Label Converter
# ---------------------------------------------------------------------------

class SanskritLabelConverter:
    """Encodes Devanagari text into token integer indices for CTC loss."""

    def __init__(self, dict_path: str | Path) -> None:
        self.dict_path = Path(dict_path)
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

        # Index 0 is reserved for CTC blank token
        self.blank_id = 0
        self.id_to_char[0] = ""

        current_id = 1
        if self.dict_path.exists():
            with open(self.dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    ch = line.strip("\r\n")
                    if ch and ch not in self.char_to_id:
                        self.char_to_id[ch] = current_id
                        self.id_to_char[current_id] = ch
                        current_id += 1

        # Space character
        if " " not in self.char_to_id:
            self.char_to_id[" "] = current_id
            self.id_to_char[current_id] = " "
            current_id += 1

        self.num_classes = current_id

    def encode(self, text: str) -> List[int]:
        """Convert a Sanskrit text string to a list of token IDs."""
        ids: List[int] = []
        for ch in text:
            if ch in self.char_to_id:
                ids.append(self.char_to_id[ch])
            elif ch != "\t" and ch != "\n":
                # Unknown characters map to blank or skip
                pass
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode greedy CTC sequence IDs to a Sanskrit string."""
        chars: List[str] = []
        prev_id = -1
        for idx in ids:
            if idx != prev_id and idx != self.blank_id:
                chars.append(self.id_to_char.get(idx, ""))
            prev_id = idx
        return "".join(chars)


# ---------------------------------------------------------------------------
# Dataset for Cropped Text Lines
# ---------------------------------------------------------------------------

class SanskritLineDataset(Dataset):
    """Paddle Dataset for loading and resizing Sanskrit line images."""

    def __init__(
        self,
        data_dir: Path,
        label_file: Path,
        converter: SanskritLabelConverter,
        target_h: int = 48,
        target_w: int = 320,
        max_len: int = 60,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.converter = converter
        self.target_h = target_h
        self.target_w = target_w
        self.max_len = max_len
        self.samples: List[Tuple[str, str]] = []

        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip("\r\n").split("\t")
                    if len(parts) >= 2:
                        rel_path, text = parts[0], parts[1]
                        full_path = self.data_dir / rel_path
                        if full_path.exists():
                            self.samples.append((str(full_path), text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.ones((self.target_h, self.target_w, 3), dtype=np.uint8) * 255

        # Resize keeping aspect ratio, padding with 255
        h, w = img.shape[:2]
        ratio = float(self.target_h) / max(1, h)
        new_w = min(self.target_w, max(1, int(w * ratio)))
        resized = cv2.resize(img, (new_w, self.target_h))

        canvas = np.ones((self.target_h, self.target_w, 3), dtype=np.uint8) * 255
        canvas[:, :new_w] = resized

        # Normalize to [-1.0, 1.0] and CHW format
        normalized = (canvas.astype(np.float32) / 127.5) - 1.0
        chw = np.transpose(normalized, (2, 0, 1))

        # Encode text labels
        token_ids = self.converter.encode(text)[:self.max_len]
        length = len(token_ids)

        # Pad tokens to max_len
        padded_ids = np.zeros((self.max_len,), dtype=np.int32)
        if length > 0:
            padded_ids[:length] = token_ids

        return (
            paddle.to_tensor(chw, dtype="float32"),
            paddle.to_tensor(padded_ids, dtype="int32"),
            paddle.to_tensor(length, dtype="int64"),
        )


# ---------------------------------------------------------------------------
# Sanskrit CRNN Recognition Architecture
# ---------------------------------------------------------------------------

class SanskritCRNN(nn.Layer):
    """Lightweight Convolutional Recurrent Network for Sanskrit OCR."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # Feature extraction (CNN)
        self.features = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),  # -> 24 x 160

            nn.Conv2D(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),  # -> 12 x 80

            nn.Conv2D(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2D(256),
            nn.ReLU(),

            nn.Conv2D(256, 256, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2D(256),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=(2, 1), stride=(2, 1)),  # -> 6 x 80

            nn.Conv2D(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2D(512),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=(2, 1), stride=(2, 1)),  # -> 3 x 80

            nn.Conv2D(512, 512, kernel_size=(3, 1), padding=0),  # -> 1 x 80
            nn.BatchNorm2D(512),
            nn.ReLU(),
        )

        # Sequence modeling (Bidirectional GRU)
        self.rnn = nn.GRU(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            direction="bidirectional",
            time_major=False,
        )

        # Output projection for CTC classes (characters + blank)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        # x: [B, 3, 48, 320]
        feats = self.features(x)  # [B, 512, 1, 80]
        feats = paddle.squeeze(feats, axis=2)  # [B, 512, 80]
        feats = paddle.transpose(feats, perm=[0, 2, 1])  # [B, 80, 512]

        rnn_out, _ = self.rnn(feats)  # [B, 80, 512]
        logits = self.fc(rnn_out)  # [B, 80, num_classes]
        return logits


# ---------------------------------------------------------------------------
# Training & Validation Runner
# ---------------------------------------------------------------------------

def train_sanskrit_ocr(
    data_dir: Path,
    dict_path: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 0.0005,
    dry_run: bool = False,
) -> None:
    """Run fine-tuning training loop on Sanskrit multi-domain lines."""
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = SanskritLabelConverter(dict_path)
    print(f"[*] Loaded Sanskrit character dictionary: {converter.num_classes} classes (including blank)")

    train_label_file = data_dir / "train_labels.txt"
    val_label_file = data_dir / "val_labels.txt"

    train_dataset = SanskritLineDataset(data_dir, train_label_file, converter)
    val_dataset = SanskritLineDataset(data_dir, val_label_file, converter)

    print(f"[*] Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    if len(train_dataset) == 0:
        print("[!] No training samples found. Run generate_sanskrit_dataset.py first.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = SanskritCRNN(num_classes=converter.num_classes)
    ctc_loss = nn.CTCLoss(blank=0, reduction="mean")
    optimizer = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())

    if dry_run:
        print("[*] Dry run mode enabled: executing single batch check...")
        for imgs, labels, lengths in train_loader:
            logits = model(imgs)
            log_probs = paddle.nn.functional.log_softmax(logits, axis=2)
            log_probs_tbc = paddle.transpose(log_probs, perm=[1, 0, 2])
            input_lengths = paddle.full([imgs.shape[0]], logits.shape[1], dtype="int64")
            loss = ctc_loss(log_probs_tbc, labels, input_lengths, lengths)
            print(f"[+] Dry run pass successful! Computed CTC loss: {loss.item():.4f}")
            break
        return

    print("=" * 65)
    print(f"[*] STARTING SANSKRIT MULTI-DOMAIN FINE-TUNING ({epochs} Epochs)")
    print("=" * 65)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for imgs, labels, lengths in train_loader:
            optimizer.clear_grad()
            logits = model(imgs)  # [B, T, NumClasses]
            log_probs = paddle.nn.functional.log_softmax(logits, axis=2)
            log_probs_tbc = paddle.transpose(log_probs, perm=[1, 0, 2])
            input_lengths = paddle.full([imgs.shape[0]], logits.shape[1], dtype="int64")

            loss = ctc_loss(log_probs_tbc, labels, input_lengths, lengths)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1

        avg_train_loss = total_loss / max(1, batches)

        # Validation pass
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with paddle.no_grad():
            for v_imgs, v_labels, v_lengths in val_loader:
                v_logits = model(v_imgs)
                v_log_probs = paddle.nn.functional.log_softmax(v_logits, axis=2)
                v_log_probs_tbc = paddle.transpose(v_log_probs, perm=[1, 0, 2])
                v_in_lengths = paddle.full([v_imgs.shape[0]], v_logits.shape[1], dtype="int64")
                v_loss = ctc_loss(v_log_probs_tbc, v_labels, v_in_lengths, v_lengths)
                val_loss += float(v_loss.item())
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches) if val_batches > 0 else avg_train_loss

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"| Train CTC Loss: {avg_train_loss:.4f} "
            f"| Val Loss: {avg_val_loss:.4f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = output_dir / "best_model.pdparams"
            paddle.save(model.state_dict(), str(best_path))

    # Save final model & metadata
    final_path = output_dir / "final_model.pdparams"
    paddle.save(model.state_dict(), str(final_path))

    meta = {
        "num_classes": converter.num_classes,
        "epochs": epochs,
        "best_val_loss": best_val_loss,
        "training_time_sec": round(time.time() - start_time, 2),
        "domains": ["digital", "handwritten", "manuscript"],
        "dict_path": str(dict_path),
    }
    with open(output_dir / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("=" * 65)
    print(f"[+] Fine-tuning complete! Models saved in: {output_dir}")
    print(f"    - Best Model:  {best_path}")
    print(f"    - Metadata:    {output_dir / 'model_meta.json'}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Sanskrit OCR Model with CTC")
    parser.add_argument("--data_dir", type=str, default="data/sanskrit_multidomain_dataset")
    parser.add_argument("--dict_path", type=str, default="data/sanskrit_dict.txt")
    parser.add_argument("--output_dir", type=str, default="models/sanskrit_finetuned")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()
    train_sanskrit_ocr(
        data_dir=Path(args.data_dir),
        dict_path=Path(args.dict_path),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dry_run=args.dry_run,
    )
