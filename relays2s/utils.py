import os
import random
import numpy as np
import torch
from multiprocessing.pool import ThreadPool
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from tqdm import tqdm
from llm_utils import build_prompt, run_llm
from pydantic import BaseModel
from transformers import LogitsProcessor, LogitsProcessorList

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

# ── Dialogue Formatting ───────────────────────────────────────────────────────

def add_spk_label(dialog_context, max_words=200, orders=["User: ", "Assistant: "]):
    if type(dialog_context) != list:
        dialog_context = dialog_context.split("\n")
    context = list(reversed(dialog_context))

    for i in range(0, len(context)):
        if i % 2 == 0:
            context[i] = orders[0] + context[i]
        else:
            context[i] = orders[1] + context[i]

        tokens = "\n".join(context[: i + 1]).split()
        if len(tokens) > max_words:
            break
    i = max(1, i)
    if len(tokens) < max_words:
        context = context[: i + 1]
    else:
        context = context[:i]

    context = list(reversed(context))
    return "\n\n".join(context)


# ── Evaluation ────────────────────────────────────────────────────────────────
class ResponseQuality(BaseModel):
    score: int

def _eval_worker(args):
    context, response, model, api_key = args
    context = add_spk_label(context)
    prompt, _ = build_prompt(prompts_path='prompts.yaml', prompt_key='response_quality', context = context, response = response)
    
    try:
        result = run_llm(
            prompt=prompt,
            model=model,
            response_format=ResponseQuality,
            api_key=api_key,
            reasoning_effort = 'high',
        )
        return result['score']
    except Exception as e:
        print(f"[WARN] Quality eval failed: {e}")
        return None

def eval_quality(contexts, responses, model, api_key=None, num_workers=6):
    args = [(c, r, model, api_key) for c, r in zip(contexts, responses)]
    with ThreadPool(processes=num_workers) as pool:
        scores = list(tqdm(
            pool.imap(_eval_worker, args),
            total=len(args),
            desc="Evaluating quality",
            unit="sample",
        ))
    return scores

# ── Local LLM ─────────────────────────────────────────────────────────────────

def load_llm(model_name: str):
    """Load LLM and tokenizer once at startup."""
    print(f"Loading LLM: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    print("LLM loaded successfully ✅")
    return model, tokenizer

def build_messages(context_list, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    for i, text in enumerate(context_list):
        messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": text})
    return messages

class TokenTimingProcessor(LogitsProcessor):
    def __init__(self, start_time: float):
        self.start_time = start_time        # set BEFORE generate() is called
        self.token_times: list[float] = []

    def reset(self, start_time: float):
        self.start_time = start_time
        self.token_times = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        self.token_times.append(time.perf_counter() - self.start_time)
        return scores

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

def llm_generate(
    model,
    tokenizer,
    contexts: list[list[str]],
    forced_prefixes: list[str] = None,
    max_new_tokens: int = 1024,
    system_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
) -> list[dict]:
    """
    Batched generation with per-sequence word-level timing.

    Each result dict contains:
        response_text        : full decoded response (prefix prepended if given)
        time_to_first_n_words: list of floats, index i = time when word (i+1) appeared
        ttfw                 : time-to-first-word (seconds)
        total_time           : time when the last token was emitted (seconds)
    """
    device = next(model.parameters()).device
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if forced_prefixes is None:
        forced_prefixes = [""] * len(contexts)
    assert len(forced_prefixes) == len(contexts)

    # ── build prompts ─────────────────────────────────────────────────────────
    prompts = []
    for context, prefix in zip(contexts, forced_prefixes):
        text = tokenizer.apply_chat_template(
            build_messages(context, system_prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
        if prefix:
            text += prefix
        prompts.append(text)

    # ── tokenize (left-pad for batched generation) ────────────────────────────
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    padded_input_len = inputs.input_ids.shape[1]   # uniform across batch

    # ── generate ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    timing_processor = TokenTimingProcessor(start_time=t0)

    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([timing_processor]),
        )

    # timing_processor.token_times is shared across all batch sequences
    # (all sequences step in lockstep during batched greedy decoding)

    # ── decode & attach timing per sequence ──────────────────────────────────
    results = []
    for i, prefix in enumerate(forced_prefixes):
        new_ids = output_ids[i, padded_input_len:]

        # strip padding tokens from this sequence
        seq_new_ids = new_ids[new_ids != tokenizer.pad_token_id]
        seq_len = len(seq_new_ids)

        response_text = tokenizer.decode(seq_new_ids, skip_special_tokens=True)
        if prefix:
            response_text = prefix + response_text

        # trim shared timing list to this sequence's actual length
        token_times = timing_processor.token_times[:seq_len]

        time_to_words = compute_time_to_words(
            seq_new_ids, token_times, tokenizer
        )
        
        results.append({
            "response_text": response_text,
            "time_to_words": time_to_words,
        })
    return results


# ── API LLM ─────────────────────────────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor
from litellm import completion

def _build_messages(context_list: list[str], system_prompt: str, forced_prefix: str = "") -> list[dict]:
    # 1. Start with the system prompt
    messages = [{"role": "system", "content": system_prompt}]
    
    # 2. If no prefix, just do standard alternating messages
    if not forced_prefix:
        for i, text in enumerate(context_list):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": text})
        return messages

    # 3. If there IS a prefix, we treat the conversation history as a "Case Study" 
    # and the very last message as the "Task"
    history_str = ""
    for i, text in enumerate(context_list):
        role = "User" if i % 2 == 0 else "Assistant"
        history_str += f"{role}: {text}\n"

    # The final User message is the ONLY one the model sees for the 'current' turn
    instruction_prompt = f"""You are continuing a conversation. 

### CONVERSATION HISTORY:
{history_str}

### TASK:
The Assistant has already started their next response. Continue the text from the exact point where the prefix ends. 
Output ONLY the continuation. No labels, no repeats, no conversational filler.

### PREFIX TO CONTINUE:
{forced_prefix}"""

    messages.append({"role": "user", "content": instruction_prompt})
    return messages

def _join_prefix_response(prefix: str, generated: str) -> str:
    if not prefix:
        return generated
    
    combined = prefix + " " + generated
    # Find and remove stuttered overlap
    prefix_words = prefix.split()
    for overlap_len in range(len(prefix_words), 0, -1):
        tail = " ".join(prefix_words[-overlap_len:])
        if generated.lstrip().startswith(tail):
            generated = generated.lstrip()[len(tail):].lstrip()
            return (prefix + " " + generated).strip()
    
    return combined.strip()

def _stream_single(
    model_name: str,
    context: list[str],
    forced_prefix: str,
    system_prompt: str,
    max_retries: int = 3,
) -> dict:
    """Stream one completion, recording per-word wallclock times."""
    for attempt in range(max_retries):
        try:
            messages = _build_messages(context, system_prompt, forced_prefix=forced_prefix)
            
            t0 = time.perf_counter()
            response = completion(
                model=model_name,
                messages=messages,
                stream=True,
                timeout=30,
                stream_timeout=60,  # kills hung chunks
            )

            full_text = ""
            word_times: list[float] = []
            prev_word_count = 0
        
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                full_text += delta
                now = time.perf_counter() - t0
        
                current_word_count = len(full_text.split())
                if current_word_count > prev_word_count:
                    for _ in range(current_word_count - prev_word_count):
                        word_times.append(round(now, 5))
                    prev_word_count = current_word_count

            response_text = _join_prefix_response(forced_prefix, full_text)
            return {"response_text": response_text, "time_to_words": word_times}

        except Exception as e:
            wait = (2 ** attempt) + random.random()
            print(f"[WARN] Attempt {attempt+1}/{max_retries}: {e}. Retry in {wait:.1f}s")
            time.sleep(wait)
    
    print("[ERROR] All retries exhausted")
    return {"response_text": "", "time_to_words": []}

def litellm_generate(
    model_name: str,
    contexts: list[list[str]],
    forced_prefixes: list[str] | None = None,
    system_prompt: str = "You are a helpful assistant.",
) -> list[dict]:
    """
    Parallel streaming generation via litellm.
 
    Returns list[dict] with keys:
        response_text  : full decoded response (prefix prepended)
        time_to_words  : list[float] — index i = wallclock (s) when word (i+1) appeared
    """
    if forced_prefixes is None:
        forced_prefixes = [""] * len(contexts)
    assert len(forced_prefixes) == len(contexts)
 
    def _worker(args):
        ctx, prefix = args
        return _stream_single(model_name, ctx, prefix, system_prompt)
    
    with ThreadPoolExecutor(max_workers=len(contexts)) as pool:
        results = list(pool.map(_worker, zip(contexts, forced_prefixes)))
    
    return results