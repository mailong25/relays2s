"""
run_eval.py

Evaluate quality scores for S2S Only / Cascaded Only / RelayS2S.

Usage:
  # S2S Only
  python run_eval.py --path_to_prediction data/s2s_gen_test.jsonl

  # Cascaded Only (local model)
  python run_eval.py --path_to_prediction data/s2s_gen_test.jsonl \
      --cascaded_llm "Qwen/Qwen2.5-7B-Instruct" \
      --cascaded_llm_backend local \
      --cascaded_replace_from_idx 0 \
      --asr_map_path data/asr_map.json \

  # Cascaded Only (litellm API)
  python run_eval.py --path_to_prediction data/s2s_gen_test.jsonl \
      --cascaded_llm "openai/gpt-4o" \
      --cascaded_llm_backend litellm \
      --cascaded_replace_from_idx 0 \
      --asr_map_path data/asr_map.json \

  # RelayS2S
  python run_eval.py --path_to_prediction data/s2s_gen_test.jsonl \
      --cascaded_llm "Qwen/Qwen2.5-7B-Instruct" \
      --cascaded_llm_backend local \
      --cascaded_replace_from_idx 5 \
      --asr_map_path data/asr_map.json \
      --num_prefix_words 5 \
      --verifier_checkpoint checkpoints/verifier/best_model.pt \
      --verifier_threshold 0.5 \
      --verifier_tokenizer checkpoints/tokenizer \
      --verifier_feature_dir data/verifier/features \
"""

import os
import json
import torch
import argparse
import time
from tqdm import tqdm
from utils import eval_quality, load_llm, llm_generate, litellm_generate

from prefix_verifier.models import PrefixGate
from transformers import AutoTokenizer
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_predictions(path, max_samples=None):
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    if max_samples:
        data = data[:max_samples]
    print(f"Loaded {len(data)} predictions from {path}")

    return data

def get_asr_result(asr_map, conv_id, real_context):
    contexts  = [item['text'] for item in asr_map[conv_id][:len(real_context)]]
    latency   = asr_map[conv_id][len(real_context) - 1]['latency']
    return contexts, latency

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _extract_ttfw(time_to_words: list[float], n: int) -> float:
    """Extract time-to-first-n-words from an existing time_to_words list."""
    if not time_to_words:
        return 0.0
    return time_to_words[min(n, len(time_to_words)) - 1]

def get_time_to_first_n_words(model, tokenizer, contexts, n = 5):
    # compute time to first n words for single item for fair comparison
    time_to_first_n_words = []
    for context in contexts:
        result = llm_generate(model, tokenizer, [context], max_new_tokens= 3 * n)
        result = result[0]['time_to_words']
        time_to_first_n_words.append(result[min(n, len(result)) - 1])
    return time_to_first_n_words

# ─────────────────────────────────────────────
# Batch processing per mode
# ─────────────────────────────────────────────

def process_batch_s2s(batch, num_prefix_words):
    responses = [item["response"] for item in batch]
    latencies = [_extract_ttfw(item['time_to_words'], num_prefix_words) for item in batch]
    return responses, latencies

def process_batch_cascaded(batch, asr_map, backend, model_or_name, tokenizer,
                           max_new_tokens, num_prefix_words):
    contexts, asr_latencies = zip(*[
        get_asr_result(asr_map, item["conv_id"], item["context"]) for item in batch
    ])

    if backend == "litellm":
        results = litellm_generate(model_or_name, contexts)
        responses = [r["response_text"] for r in results]
        gen_latencies = [_extract_ttfw(r["time_to_words"], num_prefix_words) for r in results]
    else:
        results = llm_generate(model_or_name, tokenizer, contexts, max_new_tokens=max_new_tokens)
        responses = [r["response_text"] for r in results]
        gen_latencies = get_time_to_first_n_words(model_or_name, tokenizer, contexts,
                                                  n=num_prefix_words)

    latencies = [asr + gen for asr, gen in zip(asr_latencies, gen_latencies)]
    return responses, latencies

def _load_features(feature_path):
    return torch.load(feature_path, map_location="cpu", weights_only=False)

def process_batch_relays2s(batch, asr_map, backend, model_or_name, tokenizer,
                              replace_from_idx, verifier, max_new_tokens,
                              num_prefix_words, verifier_feature_dir):
    contexts, asr_latencies = zip(*[
        get_asr_result(asr_map, item["conv_id"], item["context"]) for item in batch
    ])
    prefixes = [" ".join(item["response"].split()[:replace_from_idx]) for item in batch]
    s2s_latencies = [_extract_ttfw(item['time_to_words'], num_prefix_words) for item in batch]

    # Verifier
    verifier_latencies = []
    decisions = []
    if verifier is not None:
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            steps_features = list(executor.map(
                _load_features, [os.path.join(verifier_feature_dir, item["feature_path"]) for item in batch]
            ))
        responses = [item["response"] for item in batch]

        for feature, response in zip(steps_features, responses):
            t0 = time.perf_counter()
            decision, _ = verifier.decide(feature, response)
            decisions.append(decision)
            verifier_latencies.append(time.perf_counter() - t0)
    else:
        decisions = ["keep"] * len(batch)
        verifier_latencies = [0] * len(s2s_latencies)

    s2s_latencies = [s2s + veri for s2s, veri in zip(s2s_latencies, verifier_latencies)]

    # Cascaded generation
    effective_prefixes = [
        "" if d == "discard" else prefix
        for d, prefix in zip(decisions, prefixes)
    ]

    if backend == "litellm":
        results = litellm_generate(model_or_name, contexts, forced_prefixes=effective_prefixes)
        responses = [r["response_text"] for r in results]
        gen_latencies = [_extract_ttfw(r["time_to_words"], num_prefix_words) for r in results]
    else:
        results = llm_generate(model_or_name, tokenizer, contexts,
                               forced_prefixes=effective_prefixes,
                               max_new_tokens=max_new_tokens)
        responses = [r["response_text"] for r in results]
        gen_latencies = get_time_to_first_n_words(model_or_name, tokenizer, contexts,
                                                  n=num_prefix_words)
    
    cascaded_latencies = [asr + gen for asr, gen in zip(asr_latencies, gen_latencies)]
    latencies = [
        s2s_latencies[i] if decisions[i] == "keep" else cascaded_latencies[i]
        for i in range(len(batch))
    ]

    return responses, latencies, decisions

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    if args.cascaded_llm is None:
        mode = "s2s_only"
    elif args.cascaded_replace_from_idx == 0:
        mode = "cascaded_only"
    else:
        mode = "relays2s"

    backend = args.cascaded_llm_backend

    print(f"\n{'='*50}")
    print(f"  Mode               : {mode.upper()}")
    print(f"  cascaded_llm       : {args.cascaded_llm}")
    print(f"  llm_backend        : {backend}")
    print(f"  replace_from_idx   : {args.cascaded_replace_from_idx}")
    print(f"  verifier_checkpoint: {args.verifier_checkpoint}")
    print(f"  verifier_threshold : {args.verifier_threshold}")
    print(f"  gen_batch_size     : {args.gen_batch_size}")
    print(f"{'='*50}\n")

    data = load_predictions(args.path_to_prediction, args.max_samples)

    asr_map = {}
    if mode in ("cascaded_only", "relays2s"):
        asr_map = json.load(open(args.asr_map_path))

    # Load model: local weights or just keep the model name for litellm
    model_or_name = args.cascaded_llm
    tokenizer = None
    if mode in ("cascaded_only", "relays2s") and backend == "local":
        model_or_name, tokenizer = load_llm(args.cascaded_llm)

    # Load PrefixVerifier
    verifier = None
    if mode == "relays2s" and args.verifier_checkpoint:
        verifier_tokenizer = AutoTokenizer.from_pretrained(args.verifier_tokenizer)
        verifier = PrefixGate(
            args.verifier_checkpoint, args.verifier_threshold,
            verifier_tokenizer, args.num_prefix_words,
        )

    # Generate responses
    all_responses, all_latencies, all_decisions = [], [], []
    batches = [data[i:i + args.gen_batch_size] for i in range(0, len(data), args.gen_batch_size)]

    for batch in tqdm(batches):
        if mode == "s2s_only":
            resp, lat = process_batch_s2s(batch, args.num_prefix_words)
        elif mode == "cascaded_only":
            resp, lat = process_batch_cascaded(
                batch, asr_map, backend, model_or_name, tokenizer,
                args.max_new_tokens, args.num_prefix_words,
            )
        else:
            resp, lat, dec = process_batch_relays2s(
                batch, asr_map, backend, model_or_name, tokenizer,
                args.cascaded_replace_from_idx,
                verifier, args.max_new_tokens,
                args.num_prefix_words, args.verifier_feature_dir,
            )
            all_decisions.extend(dec)

        all_responses.extend(resp)
        all_latencies.extend(lat)

    if not all_decisions:
        all_decisions = [None] * len(all_responses)

    # Filter out empty responses (if any) before evaluation
    data, all_responses, all_latencies, all_decisions = zip(
        *[(item, r, lat, dec)
          for item, r, lat, dec in zip(data, all_responses, all_latencies, all_decisions) if r]
    )

    # Evaluate quality
    all_contexts = [item["context"] for item in data]
    print(f"\nEvaluating quality with {args.eval_model} ...")
    scores = eval_quality(
        all_contexts, all_responses,
        args.eval_model,
        api_key=args.api_key,
        num_workers=args.eval_workers,
    )

    valid_scores = [s for s in scores if s is not None]
    avg_score   = float(np.mean(valid_scores))
    low_quality_rate = len([s for s in valid_scores if s <= 3]) / len(valid_scores) * 100
    p90_latency = float(np.percentile(all_latencies, 90))
    avg_latency = float(np.mean(all_latencies))
    
    print(f"\n{'='*50}")
    print(f"  Evaluated samples  : {len(valid_scores)} / {len(scores)}")
    print(f"  Avg quality        : {avg_score:.4f}")
    print(f"  Low quality rate   : {low_quality_rate:.1f}%")
    print(f"  Avg Latency        : {avg_latency:.3f}s")
    print(f"  P90 Latency        : {p90_latency:.3f}s")
    
    if mode == "relays2s" and verifier is not None:
        n_keep = all_decisions.count("keep")
        n_discard = all_decisions.count("discard")
        total = n_keep + n_discard
        if total > 0:
            print(f"  Verifier — KEEP    : {n_keep}  ({100*n_keep/total:.1f}%)")
            print(f"  Verifier — DISCARD : {n_discard}  ({100*n_discard/total:.1f}%)")
    else:
        all_decisions = [None] * len(all_responses)

    print(f"{'='*50}\n")
    
    if args.output_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        results = [
            dict(id=item["id"], conv_id=item["conv_id"], context=item["context"],
                 response=resp, score=score, decision=dec, latency=lat)
            for item, resp, score, dec, lat in zip(data, all_responses, scores, all_decisions, all_latencies)
        ]
        with open(args.output_path, "w") as f:
            json.dump({"avg_score": avg_score, "results": results}, f, indent=2)
        print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate quality and latency for S2S Only / Cascaded Only / RelayS2S. "
                    "Mode is inferred from arguments: no --cascaded_llm → S2S Only; "
                    "--cascaded_replace_from_idx 0 → Cascaded Only; otherwise → RelayS2S.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input / output ───────────────────────────────────────────────────────
    p.add_argument("--path_to_prediction", type=str, required=True,
                   help="Path to a JSONL file of fast-path S2S generations. Each line must "
                        "contain at least: id, conv_id, context, response, time_to_words, "
                        "and feature_path.")
    p.add_argument("--asr_map_path", type=str, default="data/asr_map.json",
                   help="Path to the ASR map JSON (conv_id → list of {text, latency}). "
                        "Required for cascaded and RelayS2S modes.")
    p.add_argument("--output_path", type=str, default=None,
                   help="If set, save per-sample results (id, response, score, decision, "
                        "latency) and the aggregate avg_score to this JSON file.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Evaluate only the first N samples from --path_to_prediction. "
                        "Useful for quick debugging runs.")

    # ── Cascaded LLM ─────────────────────────────────────────────────────────
    p.add_argument("--cascaded_llm", type=str, default=None,
                   help="Slow-path LLM identifier. For local inference, a HuggingFace model "
                        "name (e.g. 'Qwen/Qwen2.5-7B-Instruct'). For API inference, a "
                        "litellm model string (e.g. 'openai/gpt-4o'). If omitted, runs in "
                        "S2S-only mode.")
    p.add_argument("--cascaded_llm_backend", type=str, default="local",
                   choices=["local", "litellm"],
                   help="How to run the cascaded LLM. 'local' loads HuggingFace weights "
                        "onto GPU; 'litellm' streams via the litellm API (supports OpenAI, "
                        "Gemini, Anthropic, etc.).")
    p.add_argument("--cascaded_replace_from_idx", type=int, default=0,
                   help="Word index at which the cascaded LLM takes over from the S2S "
                        "prefix. 0 = replace the entire response (Cascaded Only mode). "
                        "5 = keep the first 5 S2S words and continue from there (RelayS2S).")
    p.add_argument("--max_new_tokens", type=int, default=1024,
                   help="Maximum number of new tokens the cascaded LLM may generate per "
                        "response (local backend only).")
    p.add_argument("--gen_batch_size", type=int, default=8,
                   help="Number of samples processed per batch during cascaded generation.")

    # ── Prefix verifier ──────────────────────────────────────────────────────
    p.add_argument("--verifier_checkpoint", type=str, default=None,
                   help="Path to the trained prefix verifier checkpoint (.pt). If omitted "
                        "in RelayS2S mode, all prefixes are kept unconditionally.")
    p.add_argument("--verifier_threshold", type=float, default=0.5,
                   help="Verifier confidence threshold. Prefixes with predicted probability "
                        "≥ threshold are committed ('keep'); below it they are discarded "
                        "and the system falls back to the cascaded path alone.")
    p.add_argument("--verifier_tokenizer", type=str, default="checkpoints/tokenizer",
                   help="Path to the tokenizer used by the prefix verifier to compute "
                        "prefix token lengths (the S2S tokenizer with special tokens).")
    p.add_argument("--verifier_feature_dir", type=str, default="data/verifier/features",
                   help="Directory containing per-sample .pt feature files (hidden states "
                        "and logit calibration signals) referenced by feature_path in the "
                        "prediction JSONL.")
    p.add_argument("--num_prefix_words", type=int, default=5,
                   help="Number of S2S prefix words to consider for both latency measurement "
                        "(time-to-first-N-words) and verifier gating.")

    # ── Quality evaluation ───────────────────────────────────────────────────
    p.add_argument("--eval_model", type=str, default="gemini/gemini-3.1-flash-lite-preview",
                   help="LLM-as-judge model used to score response quality (1–5). Passed "
                        "to litellm, so any supported model string works.")
    p.add_argument("--api_key", type=str, default=None,
                   help="API key for the eval model. If omitted, litellm falls back to the "
                        "corresponding environment variable (e.g. GEMINI_API_KEY).")
    p.add_argument("--eval_workers", type=int, default=6,
                   help="Number of parallel threads for quality evaluation API calls.")

    main(p.parse_args())