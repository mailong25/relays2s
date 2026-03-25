"""
Prefix Gating Verifier

Architecture:
    Hidden(896) → Project(64) → ScalarGate(scalars modulate hidden)
                                        ↓
                                + PosEmbed(15, 64) → TokenFF
                                        ↓
                                AttentionPool (learned query) → LN
                                        ↓
                                    Linear(64→1) → σ

~70K params.  No self-attention — proven to overfit at this data scale.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

HIDDEN_DIM = 896
NUM_SCALAR_FEATURES = 3
MAX_SEQ_LEN = 15


class ScalarGate(nn.Module):
    """Scalars multiplicatively modulate hidden states.

    h' = h * σ(W_gate · s) + W_bias · s

    Entropy/logprob/margin tell the model HOW MUCH to trust each token,
    not WHAT it means.  Gating is the right inductive bias for that.
    The additive bypass ensures scalar info flows even when the gate saturates.
    """
    def __init__(self, hidden_dim, scalar_dim):
        super().__init__()
        self.gate = nn.Linear(scalar_dim, hidden_dim)
        self.bias = nn.Linear(scalar_dim, hidden_dim)

    def forward(self, h, s):
        return h * torch.sigmoid(self.gate(s)) + self.bias(s)


class AttentionPool(nn.Module):
    """Single learned query attends over the sequence.

    Better than mean pool because it learns WHICH tokens are diagnostic.
    Single-head is enough for S≤15.
    """
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.scale = math.sqrt(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        B, S, D = x.shape
        K = self.W_k(x)                                             # (B, S, D)
        V = self.W_v(x)                                             # (B, S, D)
        scores = torch.matmul(self.query.expand(B, -1, -1),
                              K.transpose(-2, -1)) / self.scale      # (B, 1, S)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        pooled = torch.matmul(attn, V).squeeze(1)                   # (B, D)
        return self.norm(pooled), attn.squeeze(1)                    # (B, D), (B, S)


class PrefixVerifier(nn.Module):
    def __init__(self, d_model=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(HIDDEN_DIM, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = ScalarGate(d_model, NUM_SCALAR_FEATURES)
        self.pos = nn.Embedding(MAX_SEQ_LEN, d_model)
        self.token_ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool = AttentionPool(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, hidden_states, scalar_features, mask):
        h = self.proj(hidden_states)                               # (B, S, d)
        h = self.gate(h, scalar_features)                          # (B, S, d)
        h = h + self.pos(torch.arange(h.size(1), device=h.device)) # (B, S, d)
        h = h + self.token_ff(h)                                   # residual
        pooled, attn = self.pool(h, mask)                          # (B, d)
        logits = self.head(pooled)                                 # (B, 1)
        return {
            "logits": logits,
            "probs": torch.sigmoid(logits.squeeze(-1)),
            "attn_weights": attn,
        }

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        breakdown = {}
        for name, mod in [("proj", self.proj), ("gate", self.gate),
                          ("pos", self.pos), ("token_ff", self.token_ff),
                          ("pool", self.pool), ("head", self.head)]:
            breakdown[name] = sum(p.numel() for p in mod.parameters())
        breakdown["total"] = total
        return breakdown


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha, self.gamma, self.ls = alpha, gamma, label_smoothing

    def forward(self, logits, targets):
        t = targets.float()
        if self.ls > 0:
            t = t * (1 - self.ls) + 0.5 * self.ls
        bce = F.binary_cross_entropy_with_logits(logits.squeeze(-1), t, reduction="none")
        p = torch.sigmoid(logits.squeeze(-1))
        pt = torch.where(t > 0.5, p, 1 - p)
        at = torch.where(t > 0.5, self.alpha, 1 - self.alpha)
        return (at * (1 - pt) ** self.gamma * bce).mean()


# ═══════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════

def extract_scalars_from_steps(steps_features, max_step=None):
    steps = steps_features[:max_step] if max_step is not None else steps_features
    scalars = []
    for step in steps:
        topk_logits = step["topk_logits"]
        if isinstance(topk_logits, torch.Tensor):
            topk_logits = topk_logits.float()
        scalars.append([
            float(step["entropy"]),
            float(step["logprob_chosen"]),
            float(topk_logits[0] - topk_logits[1]),
        ])
    if not scalars:
        return torch.zeros(0, NUM_SCALAR_FEATURES, dtype=torch.float32)
    return torch.tensor(scalars, dtype=torch.float32)


def prepare_sample(steps_features: list, n_prefix_tokens: int,
                   max_seq_len: int = MAX_SEQ_LEN):
    n_tok = min(n_prefix_tokens, len(steps_features), max_seq_len)
    hs = torch.zeros(max_seq_len, HIDDEN_DIM)
    sc = torch.zeros(max_seq_len, NUM_SCALAR_FEATURES)
    mask = torch.zeros(max_seq_len, dtype=torch.bool)
    if n_tok > 0:
        for j in range(n_tok):
            h = steps_features[j]["hidden_state"]
            hs[j] = h.float() if isinstance(h, torch.Tensor) else torch.from_numpy(h).float()
        sc[:n_tok] = extract_scalars_from_steps(steps_features, max_step=n_tok)
        mask[:n_tok] = True
    else:
        mask[0] = True
    return hs, sc, mask


class PrefixGate:
    def __init__(self, checkpoint_path, threshold, tokenizer,
                 max_prefix_words=5, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.tokenizer = tokenizer
        self.max_prefix_words = max_prefix_words
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("model_config", {})
        self.model = PrefixVerifier(**cfg)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()
        print(f"PrefixGate loaded from {checkpoint_path} (threshold={threshold:.3f})")

    def _prefix_token_len(self, response):
        prefix = " ".join(response.split()[:self.max_prefix_words])
        return len(self.tokenizer.encode(prefix, add_special_tokens=False))

    @torch.no_grad()
    def predict(self, steps_features_list, responses):
        batch_hs, batch_sc, batch_mask = [], [], []
        for steps, resp in zip(steps_features_list, responses):
            hs, sc, mask = prepare_sample(steps, self._prefix_token_len(resp))
            batch_hs.append(hs); batch_sc.append(sc); batch_mask.append(mask)
        hs_t = torch.stack(batch_hs).to(self.device)
        sc_t = torch.stack(batch_sc).to(self.device)
        mask_t = torch.stack(batch_mask).to(self.device)
        return self.model(hs_t, sc_t, mask_t)["probs"].cpu().numpy()

    def decide(self, steps_features, response):
        p = float(self.predict([steps_features], [response])[0])
        return ("keep" if p >= self.threshold else "discard"), p

    def decide_batch(self, steps_features_list, responses):
        probs = self.predict(steps_features_list, responses)
        return ["keep" if p >= self.threshold else "discard" for p in probs], probs