import torch, json
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset

MAX_CHOICES = 5

def label_to_index(answer_key, labels):
    answer_key = answer_key.strip()
    if answer_key in labels:
        return labels.index(answer_key)
    if answer_key.isdigit():
        return int(answer_key) - 1
    raise ValueError(f"bad answerKey {answer_key!r} for labels {labels}")

class ARCDataset(Dataset):
    def __init__(self, hf_split, contexts=None, top_k=0):
        self.rows = []
        for ex in hf_split:
            texts = list(ex["choices"]["text"])
            labels = list(ex["choices"]["label"])
            if not (2 <= len(texts) <= MAX_CHOICES):
                continue

            ctx = [""] * MAX_CHOICES
            if contexts and top_k > 0:
                per_choice = contexts.get(ex["id"])
                if per_choice:
                    ctx = [" ".join(p[:top_k]) if p else "" for p in per_choice]

            self.rows.append({
                "id": ex["id"],
                "question": ex["question"],
                "choices": texts,
                "contexts": ctx,
                "label": label_to_index(ex["answerKey"], labels),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

def make_collate_fn(tokenizer, max_len, use_context=False):
    def collate(batch):
        lefts, rights, masks, labels, ids = [], [], [], [], []
        for row in batch:
            n = len(row["choices"])
            for c in range(MAX_CHOICES):
                choice = row["choices"][c] if c < n else ""
                if use_context:
                    lefts.append(row["contexts"][c] or " ")
                    rights.append(f"{row['question']} {choice}".strip() or " ")
                else:
                    lefts.append(row["question"])
                    rights.append(choice)
            masks.append([1] * n + [0] * (MAX_CHOICES - n))
            labels.append(row["label"])
            ids.append(row["id"])

        enc = tokenizer(
            lefts, rights,
            truncation="only_first" if use_context else True,
            max_length=max_len,
            padding=True,
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
    use_ctx = args.top_k > 0
    collate = make_collate_fn(tokenizer, args.max_len, use_ctx)

    def ctx_for(config, split):
        if not use_ctx:
            return None
        path = f"{args.retrieved_dir}/{config}_{split}.json"
        return json.load(open(path))
    
    # both easy and challenge
    if args.train_on == "both":
        other = "ARC-Easy" if args.config == "ARC-Challenge" else "ARC-Challenge"
        raw_other = load_dataset(args.dataset, other)
        train_ds = ConcatDataset([
            ARCDataset(raw["train"], ctx_for(args.config, "train"), args.top_k),
            ARCDataset(raw_other["train"], ctx_for(other, "train"), args.top_k),
        ])
    else:
        train_ds = ARCDataset(raw["train"], ctx_for(args.config, "train"), args.top_k)

    datasets = {
        "train": train_ds,
        "validation": ARCDataset(raw["validation"], ctx_for(args.config, "validation"), args.top_k),
        "test": ARCDataset(raw["test"], ctx_for(args.config, "test"), args.top_k),
    }
    
    loaders = {}
    for split, shuffle in [("train", True), ("validation", False), ("test", False)]:
        ds = datasets[split]
        loaders[split] = DataLoader(
            ds,
            batch_size=args.batch_size if split == "train" else args.batch_size * 2,
            shuffle=shuffle,
            collate_fn=collate,
        )
        print(f"{split}: {len(ds)} questions")
        
    return loaders["train"], loaders["validation"], loaders["test"]

if __name__ == "__main__":
    from arguments import get_args
    args = get_args()
    train_loader, val_loader, test_loader = get_dataloaders(args)

    batch = next(iter(train_loader))
    for k, v in batch.items():
        print(k, tuple(v.shape) if torch.is_tensor(v) else type(v).__name__)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    print("\n--- decoded pair (q0, choice0) ---")
    print(tok.decode(batch["input_ids"][0][0], skip_special_tokens=False))
    print("real tokens:", batch["attention_mask"][0][0].sum().item())
    print("choice_mask[0]:", batch["choice_mask"][0].tolist())
    print("label[0]:", batch["labels"][0].item())