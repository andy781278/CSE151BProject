import random
import model
import torch
import json, os
from arguments import get_args
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import get_dataloaders

def main():
    #load args
    args = get_args()

    random.seed(args.seed)
    epochs = args.epochs

    train_loader, val_loader, test_loader = get_dataloaders(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")

    model = ARCModel(args.model_name).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # training
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch['input_ids'],batch['attention_mask'])
            loss = criterion(logits,batch['labels'])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

if __name__ == "__main__":
    main()


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