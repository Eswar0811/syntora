"""
Fine-tune google/byt5-small on Tamil → Tanglish pairs.

ByT5 advantages for this task:
  ✓ Processes raw UTF-8 bytes — Tamil Unicode handled natively
  ✓ No vocabulary mismatch for rare Tamil characters
  ✓ Character-level granularity = better phoneme mapping
  ✓ Robust to orthographic variation and informal spelling

Training data format (CSV):
  tamil,tanglish
  நான்,naan
  தமிழ்,thamizh
  ...

Run:
  python train_byt5.py --data ./data/tamil_tanglish_pairs.csv \
                       --output ./checkpoints/byt5-tamil \
                       --epochs 10 --batch 16

After training, update MODEL_ID in byt5_engine.py to your checkpoint path.
"""

import argparse
import csv
import os
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "google/byt5-small"  # ~300MB — fast to fine-tune
MAX_INPUT_LEN  = 256  # Tamil text (byte-level, so 256 bytes ≈ ~85 Tamil chars)
MAX_TARGET_LEN = 128  # Tanglish output (ASCII)


# ─────────────────────────────────────────────
# BUILT-IN SEED DATASET
# (Expand this with your own data for better accuracy)
# ─────────────────────────────────────────────

SEED_PAIRS = [
    # Pronouns
    ("நான்",          "naan"),
    ("நீ",            "nee"),
    ("அவன்",         "avan"),
    ("அவள்",         "aval"),
    ("அவர்",         "avar"),
    ("நாங்கள்",      "naangal"),
    ("நீங்கள்",      "neengal"),
    ("அவர்கள்",      "avargal"),
    # Common verbs
    ("போகிறேன்",     "pogiren"),
    ("வருகிறேன்",    "varugiren"),
    ("சாப்பிடுகிறேன்", "saapidugiren"),
    ("படிக்கிறேன்",  "padikkiren"),
    ("பேசுகிறேன்",   "pesugiren"),
    ("பார்க்கிறேன்", "paarkkiren"),
    ("தூங்குகிறேன்", "thoongugiren"),
    ("ஓடுகிறேன்",    "odugiren"),
    # Nouns
    ("தமிழ்",        "thamizh"),
    ("வீடு",         "veedu"),
    ("அம்மா",        "amma"),
    ("அப்பா",        "appa"),
    ("பெயர்",        "peyar"),
    ("ஊர்",          "oor"),
    ("நாடு",         "naadu"),
    ("மக்கள்",       "makkal"),
    ("மொழி",         "mozhi"),
    ("நேரம்",        "neram"),
    ("கடல்",         "kadal"),
    ("மலை",          "malai"),
    ("வானம்",        "vaanam"),
    ("நிலம்",        "nilam"),
    # Phrases
    ("நான் வீட்டுக்கு போகிறேன்",     "naan veettukku pogiren"),
    ("எனக்கு ஒரு சந்தேகம் உள்ளது",  "enakku oru sandhegam ulladhu"),
    ("தமிழ் மொழி மிகவும் அழகானது",  "thamizh mozhi migavum azhagaandhu"),
    ("வாழ்க தமிழ்",                   "vaazhga thamizh"),
    ("வணக்கம்",                        "vanakkam"),
    ("நன்றி",                          "nandri"),
    ("என்ன பண்றே",                     "enna pannre"),
    ("எனக்கு தெரியல",                 "enakku theriyala"),
    ("சரி",                            "sari"),
    ("இல்லை",                          "illai"),
    ("ஆமா",                            "aama"),
    ("வாங்க",                          "vaanga"),
    ("அழகான",                          "azhagaana"),
    ("பொழுது போக்கு",                  "pozhudhu pokku"),
    ("இன்று",                          "indru"),
    ("நாளை",                           "naalai"),
    ("நேற்று",                         "netru"),
    ("காலை",                           "kaalai"),
    ("மாலை",                           "maalai"),
    ("இரவு",                           "iravu"),
    # Numbers
    ("ஒன்று",   "ondru"),
    ("இரண்டு",  "irandu"),
    ("மூன்று",  "moondru"),
    ("நான்கு",  "naangu"),
    ("ஐந்து",   "aindhu"),
    # Grantha / borrowed
    ("ஜலம்",   "jalam"),
    ("ஷரீர்",  "shareer"),
    ("ஹரன்",   "haran"),
]


def load_dataset_from_csv(path: str) -> list[tuple[str,str]]:
    pairs = list(SEED_PAIRS)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["tamil"].strip(), row["tanglish"].strip()))
    return pairs


def pairs_to_hf_dataset(pairs: list[tuple[str,str]]) -> HFDataset:
    return HFDataset.from_dict({
        "tamil":    [p[0] for p in pairs],
        "tanglish": [p[1] for p in pairs],
    })


def preprocess(examples, tokenizer):
    """
    ByT5 input: byte-encoded Tamil text
    ByT5 target: byte-encoded Tanglish (ASCII)
    """
    # Prefix tells ByT5 the task during multi-task fine-tuning
    inputs = [f"tamil_to_tanglish: {t}" for t in examples["tamil"]]
    targets = examples["tanglish"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding=False,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding=False,
        )

    # Replace pad token id with -100 so loss ignores padding
    labels_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in lab]
        for lab in labels["input_ids"]
    ]
    model_inputs["labels"] = labels_ids
    return model_inputs


def compute_metrics_fn(tokenizer):
    cer_metric = evaluate.load("cer")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in labels
        labels = [
            [(l if l != -100 else tokenizer.pad_token_id) for l in lab]
            for lab in labels
        ]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"cer": round(cer, 4)}

    return compute_metrics


def train(
    data_path: Optional[str],
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 3e-4,
    warmup_steps: int = 100,
):
    logger.info(f"Loading tokenizer + model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    # Load data
    if data_path and os.path.exists(data_path):
        pairs = load_dataset_from_csv(data_path)
        logger.info(f"Loaded {len(pairs)} pairs from CSV + seed")
    else:
        pairs = list(SEED_PAIRS)
        logger.info(f"Using {len(pairs)} seed pairs (no CSV provided)")

    # Split 90/10 train/eval
    split = int(0.9 * len(pairs))
    train_ds = pairs_to_hf_dataset(pairs[:split])
    eval_ds  = pairs_to_hf_dataset(pairs[split:])

    logger.info(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # Tokenise
    tokenized_train = train_ds.map(
        lambda ex: preprocess(ex, tokenizer),
        batched=True, remove_columns=["tamil", "tanglish"],
    )
    tokenized_eval = eval_ds.map(
        lambda ex: preprocess(ex, tokenizer),
        batched=True, remove_columns=["tamil", "tanglish"],
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting ByT5 fine-tuning...")
    trainer.train()

    logger.info(f"Saving fine-tuned model to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Done! Update MODEL_ID in byt5_engine.py to use this checkpoint.")


if __name__ == "__main__":
    from typing import Optional
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default=None,              help="CSV path: tamil,tanglish columns")
    parser.add_argument("--output",  default="./checkpoints/byt5-tamil-tanglish")
    parser.add_argument("--epochs",  type=int,   default=10)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--warmup",  type=int,   default=100)
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        warmup_steps=args.warmup,
    )
