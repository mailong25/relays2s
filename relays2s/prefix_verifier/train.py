"""
Prefix Gating Verifier — Training & Evaluation

    # Train
    python -m prefix_verifier.train \
        --train_jsonl data/verifier/train.jsonl \
        --val_jsonl   data/verifier/valid.jsonl \
        --test_jsonl  data/verifier/test.jsonl \
        --feature_dir data/verifier/features \
        --tokenizer   checkpoints/tokenizer \
        --save_dir    checkpoints/verifier
    
    # Eval
    python -m prefix_verifier.train \
        --test_jsonl  data/verifier/test.jsonl \
        --feature_dir data/verifier/features \
        --tokenizer   checkpoints/tokenizer \
        --eval \
        --checkpoint  checkpoints/verifier/best_model.pt \
        --thresholds  0.25 0.5 0.75
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    classification_report, confusion_matrix,
)

from prefix_verifier.models import PrefixVerifier, FocalLoss
from prefix_verifier.dataset import load_data, build_dataloaders, PrefixVerifierDataset


# ═══════════════════════════════════════════════════════
# Weighted sampler
# ═══════════════════════════════════════════════════════

def make_weighted_sampler(dataset, bad_fraction: float = 0.3) -> WeightedRandomSampler:
    """Oversample bad prefixes to ~bad_fraction of each epoch."""
    labels = []
    for i in range(len(dataset)):
        labels.append(int(dataset[i]["labels"]))
    labels = np.array(labels)

    n_bad = (labels == 0).sum()
    n_good = (labels == 1).sum()

    w_bad = bad_fraction / max(n_bad, 1)
    w_good = (1 - bad_fraction) / max(n_good, 1)

    weights = np.where(labels == 0, w_bad, w_good)
    weights = torch.from_numpy(weights).double()

    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    print(f"  WeightedRandomSampler: {n_bad:,} bad ({n_bad/len(labels):.1%}) → "
          f"target ~{bad_fraction*100:.0f}% per epoch")
    return sampler


def build_weighted_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers,
                               bad_fraction=0.3):
    sampler = make_weighted_sampler(train_ds, bad_fraction)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"  Loaders ready — bs={batch_size}, train={len(train_loader)}, "
          f"val={len(val_loader)}, test={len(test_loader)}")
    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        h = batch["hidden_states"].to(device)
        s = batch["scalar_features"].to(device)
        m = batch["mask"].to(device)
        y = batch["labels"].to(device)
        loss = criterion(model(h, s, m)["logits"], y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_labels = [], []
    for batch in loader:
        h = batch["hidden_states"].to(device)
        s = batch["scalar_features"].to(device)
        m = batch["mask"].to(device)
        y = batch["labels"].to(device)
        out = model(h, s, m)
        total_loss += criterion(out["logits"], y).item()
        n += 1
        all_probs.append(out["probs"].cpu())
        all_labels.append(y.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    loss = total_loss / max(n, 1)
    return {
        "loss": loss,
        "auroc": roc_auc_score(labels, probs),
        "ap_bad": average_precision_score(1 - labels, 1 - probs),
        "probs": probs,
        "labels": labels,
    }


def train(model, train_loader, val_loader, device, epochs=30, lr=3e-4,
          warmup_epochs=3, weight_decay=5e-4, patience=7,
          focal_alpha=0.25, focal_gamma=2.0, label_smoothing=0.05,
          save_dir="checkpoints", model_config=None):
 
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
 
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs),
    ], milestones=[warmup_epochs])
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                          label_smoothing=label_smoothing).to(device)
 
    params = model.count_parameters()
    component_str = ", ".join(f"{k}={v:,}" for k, v in params.items() if k != "total")
    print(f"\nModel: {params['total']:,} params ({component_str})")
    print(f"Device: {device} | Epochs: {epochs} | LR: {lr} | Warmup: {warmup_epochs} | Patience: {patience}")
    print(f"Focal loss: alpha={focal_alpha}, gamma={focal_gamma}, label_smoothing={label_smoothing}")
    print(f"Weight decay: {weight_decay}\n")
 
    header = f"{'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  {'AUROC':>7}  {'AP(bad)':>7}  {'LR':>10}  {'Time':>5}"
    print(header)
    print("-" * len(header))
 
    best_auroc, best_epoch, patience_ctr, history = 0.0, 0, 0, []
 
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val = validate(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
 
        print(f"{epoch:4d}  {tr_loss:8.5f}  {val['loss']:8.5f}  {val['auroc']:7.4f}  "
              f"{val['ap_bad']:7.4f}  {lr_now:10.2e}  {time.time()-t0:4.1f}s")
 
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=val["loss"],
                            val_auroc=val["auroc"], val_ap_bad=val["ap_bad"], lr=lr_now))
 
        if val["auroc"] > best_auroc:
            best_auroc, best_epoch, patience_ctr = val["auroc"], epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                         "model_config": model_config or {},
                         "val_auroc": val["auroc"], "val_ap_bad": val["ap_bad"],
                         "history": history}, save_dir / "best_model.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\nEarly stopping at epoch {epoch}. Best: {best_auroc:.4f} (epoch {best_epoch})")
                break
 
    ckpt = torch.load(save_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\nRestored best checkpoint from epoch {best_epoch} (AUROC={best_auroc:.4f})")
 
    with open(save_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    return model, history


# ═══════════════════════════════════════════════════════
# Threshold Tuning & Evaluation
# ═══════════════════════════════════════════════════════

def tune_threshold(probs, labels):
    """Find the threshold that maximises macro F1."""
    bad_probs, good_probs = probs[labels == 0], probs[labels == 1]
    best = None

    for t in np.arange(0.01, 0.99, 0.005):
        br = float(np.mean(bad_probs < t))
        ga = float(np.mean(good_probs >= t))
        f1 = float(f1_score(labels, (probs >= t).astype(int), average="macro"))
        row = dict(threshold=round(t, 3), bad_recall=round(br, 4),
                   good_accept=round(ga, 4), false_accept=round(1 - br, 4), f1=round(f1, 4))
        if best is None or f1 > best["f1"]:
            best = row

    return best


def evaluate_at_thresholds(probs, labels, thresholds):
    bad_probs  = probs[labels == 0]
    good_probs = probs[labels == 1]

    print(f"\n{'─'*90}")
    print(f"  PERFORMANCE AT SPECIFIED THRESHOLDS")
    print(f"{'─'*90}")
    print(f"  {'Threshold':>10}  {'Bad Recall':>10}  {'Good Accept':>11}  {'False Accept':>12}  {'Overall Accept':>14}  {'Macro F1':>9}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*11}  {'─'*12}  {'─'*14}  {'─'*9}")

    rows = []
    for t in sorted(thresholds):
        br  = float(np.mean(bad_probs  < t))
        ga  = float(np.mean(good_probs >= t))
        fa  = 1.0 - br
        oa  = float(np.mean(probs >= t))
        f1  = float(f1_score(labels, (probs >= t).astype(int), average="macro"))
        print(f"  {t:>10.3f}  {br:>10.4f}  {ga:>11.4f}  {fa:>12.4f}  {oa:>14.4f}  {f1:>9.4f}")
        rows.append(dict(threshold=round(t, 3), bad_recall=round(br, 4),
                         good_accept=round(ga, 4), false_accept=round(fa, 4),
                         overall_accept=round(oa, 4), f1=round(f1, 4)))

    print(f"{'─'*90}")
    return rows


def evaluate(model, test_loader, device, save_dir="checkpoints",
             thresholds=None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    val = validate(model, test_loader, FocalLoss(), device)
    probs, labels = val["probs"], val["labels"]
    thresh = tune_threshold(probs, labels)
    preds = (probs >= thresh["threshold"]).astype(int)
    cm = confusion_matrix(labels, preds)

    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"  AUROC:               {val['auroc']:.4f}")
    print(f"  AP (bad prefix):     {val['ap_bad']:.4f}")
    print(f"\n  Threshold: {thresh['threshold']:.3f}  (best macro F1)")
    print(f"    Bad-prefix recall:    {thresh['bad_recall']:.4f}")
    print(f"    Good-prefix accept:   {thresh['good_accept']:.4f}")
    print(f"    False-accept rate:    {thresh['false_accept']:.4f}")
    print(f"    Macro F1:             {thresh['f1']:.4f}")
    print(f"\n{classification_report(labels, preds, target_names=['bad_prefix', 'good_prefix'], digits=3)}")
    print(f"  Confusion Matrix:      Pred BAD   Pred GOOD")
    print(f"    Actual BAD          {cm[0,0]:>8,}   {cm[0,1]:>8,}")
    print(f"    Actual GOOD         {cm[1,0]:>8,}   {cm[1,1]:>8,}")

    threshold_rows = []
    if thresholds:
        threshold_rows = evaluate_at_thresholds(probs, labels, thresholds)

    results = dict(
        auroc=float(val["auroc"]),
        ap_bad=float(val["ap_bad"]),
        threshold=thresh,
        confusion_matrix=cm.tolist(),
        test_bad=int(np.sum(labels == 0)),
        test_good=int(np.sum(labels == 1)),
        manual_thresholds=threshold_rows,
    )
    for name, data in [
        ("eval_results.json", results),
        ("inference_config.json", dict(
            threshold=thresh["threshold"],
            bad_recall=thresh["bad_recall"],
            good_accept=thresh["good_accept"],
            false_accept=thresh["false_accept"],
            checkpoint=str(save_dir / "best_model.pt"),
        )),
    ]:
        with open(save_dir / name, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {save_dir / name}")

    return results


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", type=str, default=None)
    p.add_argument("--val_jsonl", type=str, default=None)
    p.add_argument("--test_jsonl", type=str, required=True)
    p.add_argument("--feature_dir", type=str, required=True,
                   help="Directory containing .pt feature files")
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--max_prefix_words", type=int, default=5)
    p.add_argument("--load_workers", type=int, default=None)

    # Model
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    
    # Training
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--bad_fraction", type=float, default=0.1,
                   help="Target bad-sample fraction via weighted sampling")
    p.add_argument("--no_weighted_sampling", action="store_true")
    
    # Infrastructure
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--thresholds", type=float, nargs="+", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def build_model():
        return PrefixVerifier(d_model=args.d_model, dropout=args.dropout)

    if args.eval:
        assert args.checkpoint, "--checkpoint required for --eval"
        test_ds = PrefixVerifierDataset(args.test_jsonl, args.feature_dir, tokenizer,
                                        max_prefix_words=args.max_prefix_words,
                                        num_workers=args.load_workers)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        model = build_model()
        model.load_state_dict(torch.load(args.checkpoint, weights_only=False,
                                         map_location=device)["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
        params = model.count_parameters()
        component_str = ", ".join(f"{k}={v:,}" for k, v in params.items() if k != "total")
        print(f"Model: {params['total']:,} params ({component_str})")
        evaluate(model.to(device), test_loader, device,
                 args.save_dir, thresholds=args.thresholds)
    else:
        assert args.train_jsonl and args.val_jsonl, "--train_jsonl and --val_jsonl required"
        train_ds, val_ds, test_ds = load_data(
            args.train_jsonl, args.val_jsonl, args.test_jsonl, args.feature_dir,
            tokenizer, max_prefix_words=args.max_prefix_words, num_workers=args.load_workers)

        if args.no_weighted_sampling:
            print("  Weighted sampling DISABLED — using normal shuffle")
            train_loader, val_loader, test_loader = build_dataloaders(
                train_ds, val_ds, test_ds, args.batch_size, args.num_workers)
        else:
            train_loader, val_loader, test_loader = build_weighted_dataloaders(
                train_ds, val_ds, test_ds, args.batch_size, args.num_workers,
                bad_fraction=args.bad_fraction)

        model = build_model()
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, weights_only=False,
                                             map_location=device)["model_state_dict"])
            print(f"Resuming from: {args.checkpoint}")

        model, _ = train(model, train_loader, val_loader, device, args.epochs, args.lr,
                         args.warmup_epochs, args.weight_decay, args.patience,
                         args.focal_alpha, args.focal_gamma, args.label_smoothing,
                         args.save_dir, model_config={"d_model": args.d_model, "dropout": args.dropout})
        evaluate(model, test_loader, device,
                 args.save_dir, thresholds=args.thresholds)


if __name__ == "__main__":
    main()