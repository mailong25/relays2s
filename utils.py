import random
import numpy as np
import torch

# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)                         # Python random
    np.random.seed(seed)                      # NumPy
    torch.manual_seed(seed)                   # PyTorch (CPU)
    torch.cuda.manual_seed(seed)              # PyTorch (GPU, single)
    torch.cuda.manual_seed_all(seed)          # PyTorch (GPU, multi-GPU)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def ensure_tokenizer_special_tokens(tokenizer):
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "[EOS]"})
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")

    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    if (tokenizer.bos_token is None or
        tokenizer.bos_token == tokenizer.eos_token or
        tokenizer.bos_token == tokenizer.pad_token):
        tokenizer.add_special_tokens({"bos_token": "[BOS]"})
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")

    # Add SIL (stay silent) and BOC (beginning of backchannel) tokens
    tokenizer.add_special_tokens({"additional_special_tokens": ["[SIL]"]})
    tokenizer.sil_token = "[SIL]"
    tokenizer.sil_token_id = tokenizer.convert_tokens_to_ids("[SIL]")

    tokenizer.add_special_tokens({"additional_special_tokens": ["[BOC]"]})
    tokenizer.boc_token = "[BOC]"
    tokenizer.boc_token_id = tokenizer.convert_tokens_to_ids("[BOC]")

    tokenizer.add_special_tokens({"additional_special_tokens": ["[HOD]"]})
    tokenizer.hod_token = "[HOD]"
    tokenizer.hod_token_id = tokenizer.convert_tokens_to_ids("[HOD]")

    tokenizer.add_special_tokens({"additional_special_tokens": ["[STP]"]})
    tokenizer.stp_token = "[STP]"
    tokenizer.stp_token_id = tokenizer.convert_tokens_to_ids("[STP]")

    return tokenizer

def compute_time_to_words(
    tok_ids: torch.Tensor,
    token_times: list[float],
    tokenizer,
    max_words: int = 10,
) -> list[float]:
    word_times: list[float] = []
    prev_word_count = 0

    for i in range(len(tok_ids)):
        partial_text = tokenizer.decode(tok_ids[: i + 1], skip_special_tokens=True)
        word_count = len(partial_text.split())

        if word_count > prev_word_count:
            for _ in range(word_count - prev_word_count):
                word_times.append(round(token_times[i], 5))
            prev_word_count = word_count

    return word_times[:max_words]
