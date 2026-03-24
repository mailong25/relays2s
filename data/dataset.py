import torch
import torch.utils.data
from lhotse import CutSet, Seconds, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_custom_field
import math
import json
import random

class S2SDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        frame_length: Seconds,
        sampling_rate: int,
    ):
        self.tokenizer = tokenizer
        self.frame_length = frame_length
        self.sampling_rate = sampling_rate
        self.fbank_frame_shift = 10
        self.frame_n_steps = int((1000 / self.fbank_frame_shift) * self.frame_length)

    def __getitem__(self, cuts: CutSet) -> dict:
        # collate audios
        batch_source_feats, batch_source_feats_lens = collate_feats(cuts, self.frame_n_steps)

        # collate targets
        batch_size = batch_source_feats.size(0)
        target_max_len = int(batch_source_feats.size(1) // self.frame_n_steps)
        
        # fill targets with sil_id
        batch_target_tokens = torch.ones((batch_size, target_max_len), dtype=torch.long) * self.tokenizer.sil_token_id
        batch_target_token_lens = []
        batch_source_tokens = [] # use to store user text tokens for Speech Adapter pretraining
        batch_context_texts = [] # use to store dialog context for evaluation
        batch_conv_ids = []

        for idx, cut in enumerate(cuts):
            target_len = compute_num_frames(cut.duration, self.frame_length, self.sampling_rate)
            batch_target_token_lens.append(target_len)  
            tokens = batch_target_tokens[idx]
            source_tokens = []
            acc_contexts = [] # accumulate conversation history between assistant and user
            pos_info = {} # store information (dialog context, next_response, end frame index) of a BOS position
            
            for supervision in sorted(cut.supervisions, key=lambda s: s.start):
                if supervision.speaker == 'assistant':
                    
                    response_type = supervision.custom.get('type','')

                    # Fill in target tokens with assistant text tokens
                    text_ids = self.tokenizer.encode(supervision.text, add_special_tokens=False)
                    
                    if response_type == 'backchannel':
                        text_ids = [self.tokenizer.boc_token_id] + text_ids + [self.tokenizer.eos_token_id]
                    elif response_type == 'interrupted':
                        text_ids = [self.tokenizer.bos_token_id] + text_ids + [self.tokenizer.stp_token_id]
                    else:
                        text_ids = [self.tokenizer.bos_token_id] + text_ids + [self.tokenizer.eos_token_id]
                     
                    start_pos = int(supervision.start / self.frame_length)
                    end_pos = int(supervision.end / self.frame_length)
                    end_pos = min(end_pos, len(tokens) - 1)
                    
                    # Ensure filled tokens do not exceed allocated slots, except for backchannel bcz of its short length
                    max_len = end_pos - start_pos
                    if response_type != 'backchannel':
                        text_ids = text_ids[:max_len]
                    
                    tokens[start_pos : start_pos + len(text_ids)] = torch.tensor(text_ids, dtype=torch.long)

                    # Enforce stop token in case user interruptions
                    if response_type == 'interrupted':
                        tokens[end_pos] = self.tokenizer.stp_token_id
                    
                    # store related information for response quality evaluation
                    if response_type == 'standard':
                        pos_info[start_pos] = {'context': acc_contexts.copy(), 'supervision': supervision.to_dict(), 'end_pos': end_pos}
                    if response_type != 'backchannel':
                        acc_contexts.append(supervision.text)
                
                elif supervision.speaker == 'user':
                    # Store user text with timestamps for Speech Adapter pretraining
                    text_ids = self.tokenizer.encode(supervision.text, add_special_tokens=False)
                    text_ids = [self.tokenizer.bos_token_id] + text_ids + [self.tokenizer.eos_token_id]

                    start_pos = int(supervision.start / self.frame_length)
                    end_pos = int(supervision.end / self.frame_length)
                    end_pos = min(end_pos + 1, target_max_len)
                    
                    source_tokens.append({'start': start_pos, 'end': end_pos, 'tokens': text_ids})
                    acc_contexts.append(supervision.text)
            
            # Fill padding positions with pad_id
            conv_id = cut.id.rsplit("_", 1)[0]
            batch_conv_ids.append(conv_id)
            tokens[target_len:] = self.tokenizer.pad_token_id
            batch_context_texts.append(pos_info)
            batch_source_tokens.append(source_tokens)
        
        batch_target_token_lens = torch.tensor(batch_target_token_lens, dtype=torch.long)
        return {
            "source_feats": batch_source_feats,
            "source_feats_lens": batch_source_feats_lens,
            "source_tokens": batch_source_tokens,
            "target_tokens": batch_target_tokens,
            "target_token_lens": batch_target_token_lens,
            "context_texts": batch_context_texts,
            "conv_ids": batch_conv_ids,
        }

def collate_feats(cuts, frame_n_steps: int):
    """
    Collate fbank features so that each feature length (T) is rounded
    to the nearest multiple of frame_n_steps.
    """
    feats, feat_lens = collate_custom_field(cuts, field="fbank", pad_value = 0)
    # feats: B x T x F
    
    org_steps = feats.size(1)
    target_steps = math.ceil(org_steps / frame_n_steps) * frame_n_steps
    pad_amount = target_steps - org_steps
    
    if pad_amount > 0:
        feats = torch.nn.functional.pad(feats, (0, 0, 0, pad_amount))
    
    feat_lens = torch.ceil(feat_lens / frame_n_steps) * frame_n_steps
    feat_lens = feat_lens.to(dtype=torch.long)

    return feats, feat_lens