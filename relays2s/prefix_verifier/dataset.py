"""
Prefix Gating Verifier — Dataset

Data: train/valid/test.jsonl + per-sample .pt feature files in a feature directory.
"""

import json
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from prefix_verifier.models import extract_scalars_from_steps, HIDDEN_DIM, NUM_SCALAR_FEATURES, MAX_SEQ_LEN

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _process_one_sample(args):
    feature_path, label, prefix_token_len, max_seq_len = args
    steps = torch.load(feature_path, map_location="cpu", weights_only=False)
    if isinstance(steps, dict) and "steps_features" in steps:
        steps = steps["steps_features"]

    n_tok = min(prefix_token_len, len(steps), max_seq_len)
    hs = np.zeros((max_seq_len, HIDDEN_DIM), dtype=np.float32)
    sc = np.zeros((max_seq_len, NUM_SCALAR_FEATURES), dtype=np.float32)
    mask = np.zeros(max_seq_len, dtype=np.bool_)

    if n_tok > 0:
        for j in range(n_tok):
            h = steps[j]["hidden_state"]
            if isinstance(h, torch.Tensor):
                hs[j] = h.float().numpy()
            elif isinstance(h, np.ndarray):
                hs[j] = h.astype(np.float32)
        sc[:n_tok] = extract_scalars_from_steps(steps, max_step=n_tok).numpy()
        mask[:n_tok] = True

    return hs, sc, mask, int(label), n_tok


class PrefixVerifierDataset(Dataset):
    def __init__(self, jsonl_path, feature_dir, tokenizer, max_prefix_words=5,
                 max_seq_len=MAX_SEQ_LEN, num_workers=None):
        if num_workers is None:
            num_workers = min(cpu_count(), 16)

        feature_dir = Path(feature_dir)
        items = load_jsonl(jsonl_path)
        n = len(items)
        print(f"  {jsonl_path}: {n:,} samples")

        # Compute prefix token lengths
        print(f"    Computing prefix token lengths (first {max_prefix_words} words)...")
        prefix_lens = []
        for item in items:
            prefix = " ".join(item["response"].split()[:max_prefix_words])
            prefix_lens.append(len(tokenizer.encode(prefix, add_special_tokens=False)))

        # Read labels from inline is_sensible field
        item_labels = []
        missing = 0
        for item in items:
            if "is_sensible" in item:
                item_labels.append(int(item["is_sensible"]))
            else:
                item_labels.append(1)  # default to good
                missing += 1
        if missing > 0:
            print(f"    WARNING: {missing} samples missing 'is_sensible' field (defaulting to good)")

        # Resolve feature paths: feature_dir / feature_path
        # feature_path in JSONL is just the filename (e.g. "abc123.pt")
        feature_paths = []
        for item in items:
            fp = feature_dir / item["feature_path"]
            feature_paths.append(str(fp))

        # Filter samples with 0 prefix tokens
        valid = [i for i in range(n) if prefix_lens[i] > 0]
        if len(valid) < n:
            print(f"    Skipping {n - len(valid)} samples with 0 prefix tokens")

        worker_args = [
            (feature_paths[i], item_labels[i], prefix_lens[i], max_seq_len)
            for i in valid
        ]

        print(f"    Loading .pt files with {num_workers} workers...")
        t0 = time.time()
        if num_workers <= 1:
            results = [_process_one_sample(a) for a in worker_args]
        else:
            with Pool(num_workers) as pool:
                results = pool.map(_process_one_sample, worker_args, chunksize=64)

        self.hidden_states = torch.from_numpy(np.stack([r[0] for r in results]))
        self.scalar_features = torch.from_numpy(np.stack([r[1] for r in results]))
        self.masks = torch.from_numpy(np.stack([r[2] for r in results]))
        self.labels = torch.tensor([r[3] for r in results], dtype=torch.long)

        n_bad = (self.labels == 0).sum().item()
        n_good = (self.labels == 1).sum().item()
        avg_tok = np.mean([r[4] for r in results])
        print(f"    Done: {len(valid):,} samples in {time.time()-t0:.1f}s "
              f"({n_bad:,} bad / {n_good:,} good, avg {avg_tok:.1f} tok/sample)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "hidden_states": self.hidden_states[idx],
            "scalar_features": self.scalar_features[idx],
            "mask": self.masks[idx],
            "labels": self.labels[idx],
        }


def load_data(train_jsonl, val_jsonl, test_jsonl, feature_dir, tokenizer,
              max_prefix_words=5, max_seq_len=MAX_SEQ_LEN, num_workers=None):
    kw = dict(feature_dir=feature_dir, tokenizer=tokenizer,
              max_prefix_words=max_prefix_words,
              max_seq_len=max_seq_len, num_workers=num_workers)
    print("Loading splits...")
    train_ds = PrefixVerifierDataset(train_jsonl, **kw)
    val_ds = PrefixVerifierDataset(val_jsonl, **kw)
    test_ds = PrefixVerifierDataset(test_jsonl, **kw)
    return train_ds, val_ds, test_ds


def build_dataloaders(train_ds, val_ds, test_ds, batch_size=256, num_workers=2):
    counts = torch.bincount(train_ds.labels)
    weights = (1.0 / counts.float())[train_ds.labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, sampler=sampler, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    print(f"\n  Loaders ready — bs={batch_size}, "
          f"train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    return train_loader, val_loader, test_loader