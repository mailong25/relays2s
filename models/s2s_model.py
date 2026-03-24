import torch
import torch.nn as nn
import yaml
from transformers import AutoModelForCausalLM
import random

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from models.encoder.speech_encoder import SpeechEncoder
from models.adapter import LLMAdapter
import math

class S2SModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        sil_id: int,
        llm_name: str,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.sil_id = sil_id

        # ---------- Speech Encoder ----------
        encoder_config = yaml.safe_load(open("configs/speech_encoder.yaml"))
        self.speech_encoder = SpeechEncoder(**encoder_config)
        
        # ---------- Load Pretrained LLM ----------
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            trust_remote_code=True,
            device_map=None,
            torch_dtype=torch.bfloat16,
        )
        self.llm.resize_token_embeddings(vocab_size)
        self.llm.config.use_cache = False
        self.llm.train()
        self.llm_dim = self.llm.config.hidden_size
        
        # ---------- Adapter ----------
        self.adapter = LLMAdapter(
            enc_dim=encoder_config["overview_conf"]["encoder-output-dim"],
            llm_dim=self.llm_dim,
            reduce_factor=4,
        )
        
        init_range = math.sqrt(1.0 / self.llm_dim)
        self.adapter_special_embedding = nn.Parameter(torch.empty(self.llm_dim).uniform_(-init_range, init_range))

    def forward(self, feats, feats_lens, target_tokens=None, return_dict = False):
        """
        Args:
            feats: (B, T, 80) Fbank features
            feats_lens: (B,) original lengths before padding
            target_tokens: (B, L) token IDs (optional, Qwen tokenizer)
        Returns:
            logits: (B, L, vocab_size)
        """

        # ---------- Encode speech ----------
        enc_feats, enc_pads = self.speech_encoder(feats, feats_lens)   # (B, T/4, 1024)
        enc_feats, enc_pads = self.adapter(enc_feats, enc_pads)        # (B, T/8, adapt_dim)
        attention_mask = enc_pads.squeeze(1).long()
        speech_emb = enc_feats
        
        # # # ---------- Prepare input for Qwen ----------
        if target_tokens is not None:
            B, L = target_tokens.shape
            sil_col = torch.full((B, 1), self.sil_id, dtype=target_tokens.dtype, device=target_tokens.device)
            input_ids = torch.cat([sil_col, target_tokens[:, :-1]], dim=1)
            txt_emb = self.llm.get_input_embeddings()(input_ids)
            x = speech_emb + txt_emb
        else:
            x = speech_emb
        
        # ---------- Forward through Qwen ----------
        outputs = self.llm(
            inputs_embeds=x,
            attention_mask=attention_mask,
            use_cache=return_dict,
        )
        if not return_dict:
            return outputs.logits
        else:
            return {
                "logits": outputs.logits,
                "speech_emb": speech_emb,
                "attention_mask": attention_mask,
                "past_key_values": outputs.past_key_values,
            }

    def llm_step(self, past_key_values, chunk_feats, prev_tok, log_features = False):
        """
        Args:
            past_key_values: Cached attention keys/values from the LLM (can be None for first step)
            chunk_feats: Tensor of shape (D,) — chunk features
            prev_tok: int — previous token ID
            log_features: whether to return logging features
        Returns:
            pred_tok: int — predicted next token
            past_key_values: Updated cache from the model
            step_log (optional): dict with hidden/logit features for this decoding step
        """
        prev_tok = torch.tensor([prev_tok], dtype=torch.long, device=chunk_feats.device)  # (1,)
        prev_tok_emb = self.llm.get_input_embeddings()(prev_tok).squeeze(0)  # (D,)
        input_emb = prev_tok_emb + chunk_feats  # (D,)
        
        outputs = self.llm(
            inputs_embeds=input_emb.unsqueeze(0).unsqueeze(1),  # (1, 1, D)
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=log_features,   # <-- enable hidden state logging only when needed
            return_dict=True,
        )

        logits = outputs.logits[:, -1, :]  # (1, vocab_size)
        pred_tok = torch.argmax(logits[0], dim=-1).item()
        
        if not log_features:
            return pred_tok, outputs.past_key_values, None
        
        # ---- Compute logging features ----
        logits_1d = logits[0]
        log_probs = torch.log_softmax(logits_1d, dim=-1)
        probs = torch.softmax(logits_1d, dim=-1)

        # entropy = -sum p log p
        entropy = -(probs * log_probs).sum()

        # top-k
        k = min(10, logits_1d.size(-1))
        topk_logits, topk_ids = torch.topk(logits_1d, k=k, dim=-1)
        
        # chosen token logprob (global)
        logprob_chosen = log_probs[pred_tok]

        # hidden state for current decoding step (last layer, last token)
        # shape: (1, 1, hidden_dim) -> (hidden_dim,)
        hidden_t = outputs.hidden_states[-1][0, -1, :]
        
        step_log = {
            "token_id": int(pred_tok),
            "hidden_state": hidden_t.detach().to(torch.float16).cpu(),
            "topk_token_ids": topk_ids.detach().to(torch.int32).cpu(),
            "topk_logits": topk_logits.detach().to(torch.float16).cpu(),
            "topk_probs": probs[topk_ids].detach().to(torch.float16).cpu(),
            "entropy": float(entropy.detach().float().cpu().item()),
            "logprob_chosen": float(logprob_chosen.detach().float().cpu().item()),
        }

        return pred_tok, outputs.past_key_values, step_log
    
    def adapter_pretrain_step(self, source_feats, source_feats_lens, source_segments, max_seq_len = 128):
        """
        Pre-training step for the adapter only.
        For each speech segment, we construct a sequence:
            [speech_frames ...] [SPECIAL] [target_tokens ...] <pad> <pad> ...
        The LLM must predict the target tokens autoregressively.

        Args:
            source_feats: (B, T, 80) raw fbank features
            source_feats_lens: (B,) original lengths
            source_segments: List[B] of list of dicts
                [{'start': int, 'end': int, 'tokens': List[int]}, ...]

        Returns:
            logits: (N_total_segments, max_seq_len, vocab_size)
            loss:   scalar cross-entropy loss (ignores pad_id)
            acc:    token-level accuracy (ignores padding)
        """
        
        # 1. Encode speech → adapter output
        enc_feats, enc_pads = self.speech_encoder(source_feats, source_feats_lens)  # (B, T//4, enc_dim)
        enc_feats, _ = self.adapter(enc_feats, enc_pads)                            # (B, T//8, llm_dim)

        device = enc_feats.device
        dtype = enc_feats.dtype

        # 2. Flatten all segments across the batch
        all_segments = []
        for batch_idx, segments in enumerate(source_segments):
            for seg in segments:
                tokens = seg['tokens'][:max_seq_len]  # truncate if too long
                all_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'tokens': tokens,
                    'batch_idx': batch_idx,
                })

        N = len(all_segments)  # total number of segments in the batch

        # 3. Compute sequence lengths and allocate tensors
        seq_lengths = []
        for seg in all_segments:
            speech_len = seg['end'] - seg['start']
            text_len = len(seg['tokens'])
            seq_len = speech_len + 1 + text_len  # speech + special + text
            seq_lengths.append(seq_len)
        
        max_seq_len = max(seq_lengths)
        
        inputs_embeds   = torch.zeros(N, max_seq_len, self.llm_dim, dtype=dtype, device=device)
        target_tokens   = torch.full((N, max_seq_len), self.pad_id, dtype=torch.long, device=device)
        attention_mask  = torch.zeros(N, max_seq_len, dtype=torch.long, device=device)

        # 4. Fill tensors
        for idx, seg in enumerate(all_segments):
            start = seg['start']
            end = seg['end']
            speech_len = end - start
            tokens = torch.tensor(seg['tokens'], dtype=torch.long, device=device)

            text_start = speech_len + 1
            
            # Speech part
            inputs_embeds[idx, :speech_len] = enc_feats[seg['batch_idx'], start:end]

            # Special token
            inputs_embeds[idx, speech_len] = self.adapter_special_embedding

            # Target text part (teacher-forcing)
            if len(tokens) > 0:
                text_emb = self.llm.get_input_embeddings()(tokens)
                inputs_embeds[idx, text_start:text_start + len(tokens)] = text_emb
                target_tokens[idx, text_start:text_start + len(tokens)] = tokens

            # Attention mask (everything up to the end of real content)
            real_len = speech_len + 1 + len(tokens)
            attention_mask[idx, :real_len] = 1

        # 5. Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (N, max_seq_len, vocab_size)

        # 6. Loss & accuracy (standard LM objective)
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_tokens[:, 1:].contiguous()

        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        with torch.no_grad():
            preds = shift_logits.argmax(dim=-1)
            valid_mask = shift_labels != self.pad_id
            correct = (preds == shift_labels) & valid_mask
            acc = correct.sum().float() / valid_mask.sum().float()

        return logits, loss, acc
    
    def load_pretrained_adapter(self, ckpt_path):
        self.adapter.load_pretrained_subsamplers(ckpt_path)

    def load_pretrained_speech_encoder(self, ckpt_path):
        self.speech_encoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    
    def freeze_speech_encoder(self, num_layers = 12):
        for param in self.speech_encoder.parameters():
           param.requires_grad = False
        
        for i, layer in enumerate(self.speech_encoder.enc[1].encoders):
            if i >= num_layers:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def enable_gradient_checkpointing(self):
        self.llm.gradient_checkpointing_enable()
        self.llm.config.use_cache = False
        encoders = self.speech_encoder.enc[1].encoders
        for i, layer in enumerate(encoders):
            encoders[i] = checkpoint_wrapper(layer)