import random, torch, json, os, matplotlib, statistics

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from model import ARCModel
from evaluate import evaluate
from arguments import get_args
from collections import defaultdict
from dataloader import get_dataloaders
from transformers import get_linear_schedule_with_warmup


def run_seed(args, seed, loaders, device):
    train_loader, val_loader, test_loader = loaders

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ARCModel(args.model_name, args.grad_ckpt).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # warm up
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    
    # training
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val, best_state = -1.0, None
    
    for epoch in range(args.epochs):
        # training pass
        model.train()
        running = 0.0
        correct,seen = 0,0
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            choice_mask = batch["choice_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask, choice_mask)
                loss = criterion(logits,labels)
            
            running += loss.item() * labels.size(0)
            correct += (logits.argmax(-1) == labels).sum().item()
            seen += labels.size(0)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        train_losses.append(running / len(train_loader.dataset))
        train_accs.append(correct / seen)
        
        # evaluating pass
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # print info
        print(f"epoch {epoch}  train {train_losses[-1]:.4f}  "
              f"val {val_losses[-1]:.4f}  acc {val_acc:.4f}")

        # keep track best one for testing
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss, test_acc, records = evaluate(model, test_loader, criterion, device)
    print(f"  seed {seed}  best val {best_val:.4f}  test {test_acc:.4f}\n")

    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    return {"seed": seed, "best_val_acc": best_val, "test_acc": test_acc,
            "train_losses": train_losses, "val_losses": val_losses,
            "train_accs": train_accs, "val_accs": val_accs,
            "records": records}


def ensemble(runs):
    sums, gold = defaultdict(lambda: None), {}
    for r in runs:
        for rec in r["records"]:
            qid = rec["id"]
            gold[qid] = rec["gold"]
            p = rec["probs"]
            sums[qid] = p if sums[qid] is None else [a + b for a, b in zip(sums[qid], p)]

    correct = sum(1 for qid, s in sums.items()
                  if max(range(len(s)), key=lambda i: s[i]) == gold[qid])
    return correct / len(sums)

    
if __name__ == "__main__":
    #configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")
    
    #load args
    args = get_args()
    loaders = get_dataloaders(args)
    
    src = os.path.basename(args.retrieved_dir.rstrip("/"))
    seed_str = "-".join(str(s) for s in sorted(args.seeds))
    tag = f"{args.config}_{args.train_on}_k{args.top_k}_{src}_seeds{seed_str}"

    runs = [run_seed(args, s, loaders, device) for s in args.seeds]

    accs = [r["test_acc"] for r in runs]
    mean = statistics.mean(accs)
    std = statistics.stdev(accs) if len(accs) > 1 else 0.0
    ens = ensemble(runs) if len(runs) > 1 else accs[0]
    
    print(f"\n{tag}")
    print(f"  per seed : {['%.4f' % a for a in accs]}")
    print(f"  mean     : {mean:.4f} +/- {std:.4f}")
    print(f"  ensemble : {ens:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/{tag}_results.json", "w") as f:
        json.dump({
            "args": vars(args),
            "per_seed": [{k: v for k, v in r.items() if k != "records"} for r in runs],
            "mean_test_acc": mean, "std_test_acc": std, "ensemble_test_acc": ens,
        }, f, indent=2)

    with open(f"{args.out_dir}/{tag}_preds.json", "w") as f:
        json.dump({str(r["seed"]): r["records"] for r in runs}, f, indent=2)

    os.makedirs("plots", exist_ok=True)
    xs = range(1, args.epochs + 1)
    
    # loss plot
    plt.figure(figsize=(6, 4))
    for r in runs:
        plt.plot(xs, r["train_losses"], marker="o", alpha=0.7,
                 label=f"train s{r['seed']}")
        plt.plot(xs, r["val_losses"], marker="s", ls="--", alpha=0.7,
                 label=f"val s{r['seed']}")
    plt.axhline(1.3863, ls=":", c="gray", lw=1, label="random (ln 4)")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(tag)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"plots/{tag}_loss.png", dpi=150)
    plt.close()
    
    # acc plot
    plt.figure(figsize=(6, 4))
    for r in runs:
        plt.plot(xs, r["train_accs"], marker="o", alpha=0.7, label=f"train s{r['seed']}")
        plt.plot(xs, r["val_accs"], marker="s", ls="--", alpha=0.7, label=f"val s{r['seed']}")
    plt.axhline(0.25, ls=":", c="gray", lw=1, label="random (ln 4)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(tag)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"plots/{tag}_acc.png", dpi=150)
    plt.close()
    
    '''
    correct = 0
    for row in test:
        labels = row["choices"]["label"]
        guess = random.choice(labels)
        if guess == row["answerKey"]:
            correct += 1

    acc = correct / len(test)

    print(f"Random baseline: {acc:.3f} on {len(test)} questions")
    '''