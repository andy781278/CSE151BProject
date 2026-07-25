import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

MAX_CHOICES = 5

def label_to_index(answer_key, labels):
    answer_key = answer_key.strip()
    if answer_key in labels:
        return labels.index(answer_key)
    if answer_key.isdigit():
        return int(answer_key) - 1
    raise ValueError(f"bad answerKey {answer_key!r} for labels {labels}")

class ARCDataset(Dataset):
    def __init__(self, hf_split):
        self.rows = []
        for ex in hf_split:
            texts = list(ex["choices"]["text"])
            labels = list(ex["choices"]["label"])
            if not (2 <= len(texts) <= MAX_CHOICES):
                continue
            self.rows.append({
                "id": ex["id"],
                "question": ex["question"],
                "choices": texts,
                "label": label_to_index(ex["answerKey"], labels),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

def make_collate_fn(tokenizer, max_len):
    def collate(batch):
        questions, choices, masks, labels, ids = [], [], [], [], []

        for row in batch:
            n = len(row["choices"])
            for c in range(MAX_CHOICES):
                questions.append(row["question"])
                choices.append(row["choices"][c] if c < n else "")
            masks.append([1] * n + [0] * (MAX_CHOICES - n))
            labels.append(row["label"])
            ids.append(row["id"])

        enc = tokenizer(
            questions,
            choices,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )

        bsz = len(batch)
        out = {k: v.view(bsz, MAX_CHOICES, -1) for k, v in enc.items()}
        out["choice_mask"] = torch.tensor(masks, dtype=torch.bool)
        out["labels"] = torch.tensor(labels, dtype=torch.long)
        out["ids"] = ids
        return out

    return collate

def get_dataloaders(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    raw = load_dataset(args.dataset, args.config)
    collate = make_collate_fn(tokenizer, args.max_len)

    loaders = {}
    for split, shuffle in [("train", True), ("validation", False), ("test", False)]:
        ds = ARCDataset(raw[split])
        loaders[split] = DataLoader(
            ds,
            batch_size=args.batch_size if split == "train" else args.batch_size * 2,
            shuffle=shuffle,
            collate_fn=collate,
        )
        print(f"{split}: {len(ds)} questions")

    return loaders["train"], loaders["validation"], loaders["test"]