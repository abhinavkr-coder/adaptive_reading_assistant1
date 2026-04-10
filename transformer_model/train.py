"""Fine-tunes T5-small for sentence simplification. Run after data_builder.py."""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
)
from dataset import SimplificationDataset

MODEL_NAME = "t5-small"
OUTPUT_DIR = "./t5-simplifier"


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_ds = SimplificationDataset(tokenizer, "train")
    val_ds   = SimplificationDataset(tokenizer, "val")
    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),   # mixed precision on T4
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved → {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()