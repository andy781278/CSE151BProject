import torch

def evaluate(model, loader, criterion, device):
    model.eval()
    running, correct, total = 0.0, 0, 0
    records = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            choice_mask = batch["choice_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask, choice_mask)
                loss = criterion(logits, labels)

            probs = torch.softmax(logits.float(), dim=-1)
            
            pred = logits.argmax(-1)

            running += loss.item() * labels.size(0)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            for i, qid in enumerate(batch["ids"]):
                records.append({
                    "id": qid,
                    "pred": int(pred[i]),
                    "gold": int(labels[i]),
                    "correct": bool(pred[i] == labels[i]),
                    "probs": [round(float(x), 6) for x in probs[i]],
                })
                
    return running / total, correct / total, records