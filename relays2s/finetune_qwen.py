"""
Full fine-tune Qwen2.5-0.5B-Instruct on multi-turn conversation data (convs.jsonl).

Requirements:
    pip install torch transformers datasets trl accelerate

Usage:
    python finetune_qwen.py --model_name Qwen/Qwen2.5-0.5B-Instruct --data_path data/llm_fine_tuning_data.jsonl --output_dir checkpoints/qwen2.5-0.5b-finetuned
"""

import json
import argparse

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--data_path", type=str, default="./convs.jsonl")
    p.add_argument("--output_dir", type=str, default="./qwen2.5-0.5b-finetuned")
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--max_seq_length", type=int, default=5096)
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--system_prompt",
        type=str,
        default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    )
    return p.parse_args()


def load_conversations(data_path: str, system_prompt: str) -> list[list[dict]]:
    """
    Each line in convs.jsonl is a JSON array of turns:
        [{"role": "user", "text": "..."}, {"role": "assistant", "text": "..."}, ...]

    Convert to Qwen chat format:
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
    """
    conversations = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            turns = json.loads(line)
            messages = [{"role": "system", "content": system_prompt}]
            for turn in turns:
                messages.append({
                    "role": turn["role"],
                    "content": turn["text"],
                })
            conversations.append(messages)
    print(f"Loaded {len(conversations)} conversations from {data_path}")
    return conversations


def build_dataset(conversations: list[list[dict]], tokenizer) -> Dataset:
    texts = []
    for messages in conversations:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def main():
    args = parse_args()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model — full parameters, bf16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Data
    conversations = load_conversations(args.data_path, args.system_prompt)
    dataset = build_dataset(conversations, tokenizer)

    if args.val_split > 0 and len(dataset) > 10:
        split = dataset.train_test_split(test_size=args.val_split, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Train: {len(train_dataset)}, Val: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        dataset_text_field="text",
        packing=True,
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Trainer — no peft_config, all parameters are trained
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting full fine-tuning...")
    trainer.train()

    # Save full model + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()