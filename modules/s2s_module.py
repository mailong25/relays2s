import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
import time
import pickle
import os
import json
from transformers.cache_utils import DynamicCache
from utils import compute_time_to_words
from metrics import compute_turn_taking_metrics
import numpy as np
import json
import uuid
import shutil

def get_past_key_values(past_key_values: DynamicCache, b_idx: int, s_idx: int):
    if s_idx == 0 or past_key_values is None:
        return None
    new_cache = DynamicCache()
    for layer_idx, layer in enumerate(past_key_values.layers):
        key = layer.keys
        value = layer.values
        k_slice = key[b_idx:b_idx+1, ..., :s_idx, :].contiguous()
        v_slice = value[b_idx:b_idx+1, ..., :s_idx, :].contiguous()
        new_cache.update(k_slice, v_slice, layer_idx)
    return new_cache

class S2SLightningModule(pl.LightningModule):
    def __init__(self, model, tokenizer, cfg):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg

        weights = torch.ones(len(self.tokenizer)).to(self.model.llm.device)
        weights[self.tokenizer.sil_token_id] = 0.2
        self.loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=self.tokenizer.pad_token_id)

        # Tolerance window for turn-taking metrics
        self.turn_taking_tolerance = cfg.get('turn_taking_tolerance', 1)
        self.turn_taking_stats = None
    
    def forward(self, source_audio, source_audio_lens, target_tokens):
        return self.model(source_audio, source_audio_lens, target_tokens)
    
    def training_step(self, batch, batch_idx):
        logits, loss, acc = self.compute_step(batch)
        bs = logits.size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def on_validation_epoch_start(self):
        self.turn_taking_stats = {
            'start_speaking': {'TP': 0, 'FP': 0, 'FN': 0},
            'stop_speaking': {'TP': 0, 'FP': 0, 'FN': 0},
            'backchannel': {'TP': 0, 'FP': 0, 'FN': 0},
            'silent': {'TP': 0, 'FP': 0, 'FN': 0},
        }

        run_dir = os.path.join(self.cfg.training.checkpoint_dir, self.cfg.wandb.run_name)
        self.generation_path = os.path.join(run_dir, "generations.jsonl")
        self.feature_dir = os.path.join(run_dir, "features")
        for path in [self.generation_path, self.feature_dir]:
            if os.path.exists(path):
                print(f"Warning: {path} already exists. It will be overwritten.")
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        os.makedirs(self.feature_dir, exist_ok=True)
        
        self.total_gen_samples = 0

    def on_validation_epoch_end(self):
        for cat in ['start_speaking', 'silent', 'stop_speaking', 'backchannel']:
            for metric in ["TP", "FP", "FN"]:
                val = torch.tensor(self.turn_taking_stats[cat][metric], device=self.device)
                synced_val = self.all_gather(val).sum().item()
                self.turn_taking_stats[cat][metric] = synced_val

            # compute metrics
            tp = self.turn_taking_stats[cat]["TP"]
            fp = self.turn_taking_stats[cat]["FP"]
            fn = self.turn_taking_stats[cat]["FN"]

            pred_pos = tp + fp
            gt_pos = tp + fn

            precision = (tp / pred_pos) if pred_pos > 0 else 0.0
            recall = (tp / gt_pos) if gt_pos > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            self.log(f"val_{cat}_p", precision, prog_bar=False)
            self.log(f"val_{cat}_r", recall, prog_bar=False)
            self.log(f"val_{cat}_f1", f1, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        logits, loss, acc = self.compute_step(batch)
        bs = logits.size(0)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        
        # Compute turn-taking metrics
        batch_metrics = compute_turn_taking_metrics(
            logits=logits,
            target_tokens=batch["target_tokens"],
            target_token_lens=batch["target_token_lens"],
            tokenizer=self.tokenizer,
            tolerance_window=self.turn_taking_tolerance
        )

        for cat in ['start_speaking', 'stop_speaking', 'backchannel', 'silent']:
            self.turn_taking_stats[cat]["TP"] += batch_metrics[cat]["TP"]
            self.turn_taking_stats[cat]["FP"] += batch_metrics[cat]["FP"]
            self.turn_taking_stats[cat]["FN"] += batch_metrics[cat]["FN"]
        
        if self.cfg.eval.max_gen_samples != -1 and self.total_gen_samples >= self.cfg.eval.max_gen_samples:
            return
        
        outputs = self.model(batch["source_feats"],
                             batch["source_feats_lens"],
                             batch["target_tokens"],
                             return_dict=True)
        
        target_tokens = batch["target_tokens"]
        B = target_tokens.size(0)
        batch_preds = []
        batch_features = []
        
        for i in range(B):
            pos_info = batch["context_texts"][i]
            for bos_pos in pos_info:
                prev_tok = self.tokenizer.bos_token_id
                past_key_values = get_past_key_values(outputs["past_key_values"], i, bos_pos + 1)
                generated_ids = []
                steps_features = []
                token_times = []
                t0 = time.perf_counter()

                chunk_feats = torch.zeros_like(outputs["speech_emb"][i, 0, :]) # zeroing user speech features
                
                for j in range(1, self.cfg.eval.max_gen_tokens_per_sample):
                    is_log_feature = self.cfg.eval.log_features and (j <= 20) # Only log features for the first 20 tokens to save space
                    prev_tok, past_key_values, step_features = self.model.llm_step(past_key_values, chunk_feats, prev_tok, is_log_feature)
                    
                    token_times.append(time.perf_counter() - t0)

                    generated_ids.append(prev_tok)
                    steps_features.append(step_features)
                    if prev_tok in [self.tokenizer.eos_token_id, self.tokenizer.sil_token_id]:
                        break
                
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                context = pos_info[bos_pos]['context']
                time_to_words = compute_time_to_words(generated_ids, token_times, self.tokenizer)
                
                batch_preds.append({
                    "id": uuid.uuid4().hex,
                    "conv_id": batch["conv_ids"][i],
                    "context": context,
                    "response": response,
                    "feature_path": None,
                    "time_to_words": time_to_words
                })
                
                batch_features.append(steps_features)
        
        if self.cfg.eval.log_features:
            for pred, feats in zip(batch_preds, batch_features):
                path = os.path.abspath(os.path.join(self.feature_dir, f"{pred['id']}.pt"))
                torch.save(feats, path)
                pred["feature_path"] = path
        
        with open(self.generation_path, "a") as f:
            for pred in batch_preds:
                f.write(json.dumps(pred) + "\n")
        
        self.total_gen_samples += len(batch_preds)
    
    def compute_step(self, batch):
        source_feats = batch["source_feats"]
        source_feats_lens = batch["source_feats_lens"]
        target_tokens = batch["target_tokens"]
        logits = self(source_feats, source_feats_lens, target_tokens)

        loss  = self.loss_fn(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
        preds = logits.argmax(dim=-1)
        
        ignore_mask = (target_tokens != self.tokenizer.pad_token_id) & (target_tokens != self.tokenizer.sil_token_id)
        correct = (preds == target_tokens) & ignore_mask
        acc = correct.sum().float() / ignore_mask.sum()
        
        return logits, loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.training.learning_rate)
        return optimizer

class AdapterLightningModule(pl.LightningModule):
    def __init__(self, model, tokenizer, cfg):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg

        weights = torch.ones(len(self.tokenizer)).to(self.model.llm.device)
        weights[self.tokenizer.sil_token_id] = 0.2
        self.loss_fn = nn.CrossEntropyLoss(weight=weights, ignore_index=self.tokenizer.pad_token_id)
    
    def forward(self, source_audio, source_audio_lens, source_segments):
        return self.model.adapter_pretrain_step(source_audio, source_audio_lens, source_segments, self.cfg.training.get("max_seq_len", 128))
    
    def training_step(self, batch, batch_idx):
        logits, loss, acc = self.compute_step(batch)
        bs = batch["source_feats"].size(0)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss, acc = self.compute_step(batch)
        bs = batch["source_feats"].size(0)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
    
    def compute_step(self, batch):
        logits, loss, acc = self.model.adapter_pretrain_step(
            batch["source_feats"],
            batch["source_feats_lens"],
            batch["source_tokens"],
            self.cfg.training.get("max_seq_len", 128)
        )
        return logits, loss, acc
    
    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]        
        optimizer = torch.optim.Adam(trainable_params, lr=self.cfg.training.learning_rate)
        return optimizer
    