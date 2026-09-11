import argparse
import os
import random

import torch
import torch.nn as nn

import match_dataset as md
import match_model as mm

BASE = os.path.dirname(os.path.abspath(__file__))


def split_matches(dataset, val_frac=0.1, seed=42):
    """按局切分（不按帧）。返回 (train_idxs, val_idxs)。"""
    idxs = list(range(len(dataset)))
    rng = random.Random(seed)
    n_val = max(1, int(len(idxs) * val_frac)) if (val_frac > 0 and idxs) else 0
    val = set(rng.sample(idxs, n_val)) if n_val else set()
    train = [i for i in idxs if i not in val]
    return train, sorted(val)


def make_model():
    return mm.MatchGRU()


def make_training(device):
    model = mm.MatchGRU().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    return model, opt, crit


def train_one_epoch(model, dataset, train_idxs, opt, criterion,
                    batch_size=4, device=torch.device("cpu")):
    model.train()
    idxs = list(train_idxs)
    random.shuffle(idxs)
    total = 0.0
    count = 0
    for b in range(0, len(idxs), batch_size):
        sub = [md.truncated_item(dataset, i) for i in idxs[b:b + batch_size]]
        padded, lengths, labels = md.collate_fn(sub)
        padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)
        opt.zero_grad()
        logits = model.forward(padded, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


def evaluate(model, dataset, val_idxs, device=torch.device("cpu")):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i in val_idxs:
            feats, label = md.truncated_item(dataset, i, k=0)  # 完整序列评估
            padded, lengths, labels = md.collate_fn([(feats, label)])
            logits = model.forward(padded.to(device), lengths.to(device))
            loss_sum += crit(logits, labels.to(device)).item()
            pred = logits.argmax(dim=1).item()
            correct += (pred == label)
            total += 1
    return correct / max(total, 1), loss_sum / max(total, 1)


def train(videos_path=md.DEFAULT_VIDEOS, out_ckpt="models/match_gru.pt",
          epochs=30, seed=42, device=None):
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(seed)
    random.seed(seed)
    dataset = md.MatchDataset(videos_path)
    if len(dataset) == 0:
        raise RuntimeError(f"no labeled matches in {videos_path}")
    train_idxs, val_idxs = split_matches(dataset, val_frac=0.1, seed=seed)
    model, opt, criterion = make_training(device)
    best_acc = -1.0
    for ep in range(epochs):
        tr_loss = train_one_epoch(model, dataset, train_idxs, opt, criterion,
                                  device=device)
        val_acc, val_loss = evaluate(model, dataset, val_idxs, device=device)
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(os.path.abspath(out_ckpt)), exist_ok=True)
            torch.save(model.state_dict(), out_ckpt)
        print(f"epoch {ep}: train_loss={tr_loss:.4f} val_acc={val_acc:.3f} val_loss={val_loss:.4f}")
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default=md.DEFAULT_VIDEOS)
    parser.add_argument("--out", default=os.path.join(BASE, "models", "match_gru.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    train(args.videos, args.out, args.epochs)


if __name__ == "__main__":
    main()
