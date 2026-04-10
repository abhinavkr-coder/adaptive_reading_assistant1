"""PyTorch Dataset — reads from the JSONL files produced by data_builder.py."""

import json
from pathlib import Path
from torch.utils.data import Dataset


class SimplificationDataset(Dataset):
    def __init__(self, tokenizer, split: str = "train", max_len: int = 256,
                 data_dir: str = "./training_data"):
        path = Path(data_dir) / f"{split}.jsonl"
        with open(path) as f:
            self.pairs = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        enc = self.tokenizer(
            f"simplify: {pair['source']}",
            text_target=pair["target"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = enc["labels"].squeeze()
        # T5 loss ignores positions labelled -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         labels,
        }