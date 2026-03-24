import torch
import numpy as np
from typing import Dict, List, Tuple, Set

class TurnTakingMetrics:
    """
    Compute turn-taking metrics including start speaking, stay silent, 
    stop speaking, and backchannel predictions.
    """
    
    def __init__(self, tokenizer, tolerance_window: int = 1):
        """
        Args:
            tokenizer: Tokenizer with special tokens (bos, eos, stp, boc, sil)
            tolerance_window: Frame tolerance for matching predictions to ground truth
        """
        self.tokenizer = tokenizer
        self.tolerance = tolerance_window
        
        # Token IDs
        self.bos_id = tokenizer.bos_token_id  # Start speaking
        self.eos_id = tokenizer.eos_token_id  # End normally
        self.stp_id = tokenizer.stp_token_id  # Stop (interrupted)
        self.boc_id = tokenizer.boc_token_id  # Backchannel
        self.sil_id = tokenizer.sil_token_id  # Silent
        self.pad_id = tokenizer.pad_token_id  # Padding
    
    def extract_events(self, tokens: torch.Tensor) -> Dict[str, Set[int]]:
        """
        Extract turn-taking events from a token sequence.
        
        Args:
            tokens: (L,) tensor of token IDs
            
        Returns:
            Dictionary with sets of frame indices for each event type:
                - 'start_speaking': indices where BOS appears
                - 'stop_speaking': indices where STP appears
                - 'backchannel': indices where BOC appears
                - 'silent': indices where SIL appears
        """
        tokens = tokens.cpu().numpy()
        
        events = {
            'start_speaking': set(),
            'stop_speaking': set(),
            'backchannel': set(),
            'silent': set()
        }
        
        for idx, token in enumerate(tokens):
            if token == self.bos_id:
                events['start_speaking'].add(idx)
            elif token == self.stp_id:
                events['stop_speaking'].add(idx)
            elif token == self.boc_id:
                events['backchannel'].add(idx)
            elif token == self.sil_id:
                events['silent'].add(idx)
        
        return events
    
    def match_predictions_to_gt(
        self, 
        pred_indices: Set[int], 
        gt_indices: Set[int]
    ) -> Tuple[int, int, int]:
        """
        Match predicted indices to ground truth within tolerance window.
        Uses one-to-one matching: each GT can match at most one prediction.
        
        Args:
            pred_indices: Set of predicted event indices
            gt_indices: Set of ground truth event indices
            
        Returns:
            (TP, FP, FN) counts
        """
        matched_gt = set()
        matched_pred = set()
        
        # Sort for consistent matching
        sorted_pred = sorted(pred_indices)
        sorted_gt = sorted(gt_indices)
        
        # Match predictions to ground truth
        for p in sorted_pred:
            for g in sorted_gt:
                if g in matched_gt:
                    continue
                if abs(p - g) <= self.tolerance:
                    matched_pred.add(p)
                    matched_gt.add(g)
                    break
        
        TP = len(matched_pred)
        FP = len(pred_indices) - len(matched_pred)
        FN = len(gt_indices) - len(matched_gt)
        
        return TP, FP, FN
    
    def compute_silent_metrics(
        self,
        pred_events: Dict[str, Set[int]],
        gt_events: Dict[str, Set[int]],
        seq_length: int
    ) -> Tuple[int, int, int]:
        """
        Compute metrics for staying silent (floor-keeping).
        
        Args:
            pred_events: Predicted events dictionary
            gt_events: Ground truth events dictionary
            seq_length: Total sequence length (excluding padding)
            
        Returns:
            (TP, FP, FN) for silent predictions
        """
        pred_silent = pred_events['silent']
        gt_silent = gt_events['silent']
        
        # All non-silent predictions
        pred_non_silent = set()
        for event_type in ['start_speaking', 'stop_speaking', 'backchannel']:
            pred_non_silent.update(pred_events[event_type])
        
        # All non-silent ground truth
        gt_non_silent = set()
        for event_type in ['start_speaking', 'stop_speaking', 'backchannel']:
            gt_non_silent.update(gt_events[event_type])
        
        # TP: Predicted silent AND GT is silent
        TP = len(pred_silent & gt_silent)
        
        # FP: Predicted silent BUT GT is non-silent
        FP = len(pred_silent & gt_non_silent)
        
        # FN: Predicted non-silent BUT GT is silent
        FN = len(pred_non_silent & gt_silent)
        
        return TP, FP, FN
    
    def compute_metrics(
        self,
        pred_tokens: torch.Tensor,
        gt_tokens: torch.Tensor,
        token_lens: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute all turn-taking metrics for a batch.
        
        Args:
            pred_tokens: (B, L) predicted token IDs
            gt_tokens: (B, L) ground truth token IDs
            token_lens: (B,) actual sequence lengths (excluding padding)
            
        Returns:
            Dictionary of metrics with precision, recall, F1 for each category
        """
        batch_size = pred_tokens.size(0)
        
        # Accumulators for each metric category
        metrics_accum = {
            'start_speaking': {'TP': 0, 'FP': 0, 'FN': 0},
            'stop_speaking': {'TP': 0, 'FP': 0, 'FN': 0},
            'backchannel': {'TP': 0, 'FP': 0, 'FN': 0},
            'silent': {'TP': 0, 'FP': 0, 'FN': 0},
        }
        
        for b in range(batch_size):
            seq_len = token_lens[b].item()
            
            # Extract events from predictions and ground truth
            pred_events = self.extract_events(pred_tokens[b, :seq_len])
            gt_events = self.extract_events(gt_tokens[b, :seq_len])
            
            # 1. Start Speaking
            TP, FP, FN = self.match_predictions_to_gt(
                pred_events['start_speaking'],
                gt_events['start_speaking']
            )
            metrics_accum['start_speaking']['TP'] += TP
            metrics_accum['start_speaking']['FP'] += FP
            metrics_accum['start_speaking']['FN'] += FN
            
            # 2. Stay Silent
            TP, FP, FN = self.compute_silent_metrics(
                pred_events, gt_events, seq_len
            )
            metrics_accum['silent']['TP'] += TP
            metrics_accum['silent']['FP'] += FP
            metrics_accum['silent']['FN'] += FN
            
            # 3. Stop Speaking
            TP, FP, FN = self.match_predictions_to_gt(
                pred_events['stop_speaking'],
                gt_events['stop_speaking']
            )
            metrics_accum['stop_speaking']['TP'] += TP
            metrics_accum['stop_speaking']['FP'] += FP
            metrics_accum['stop_speaking']['FN'] += FN
            
            # 4. Backchannel
            TP, FP, FN = self.match_predictions_to_gt(
                pred_events['backchannel'],
                gt_events['backchannel']
            )
            metrics_accum['backchannel']['TP'] += TP
            metrics_accum['backchannel']['FP'] += FP
            metrics_accum['backchannel']['FN'] += FN

        return metrics_accum

def compute_turn_taking_metrics(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    target_token_lens: torch.Tensor,
    tokenizer,
    tolerance_window: int = 1
) -> Dict[str, float]:
    """
    Convenience function to compute turn-taking metrics from model outputs.
    
    Args:
        logits: (B, L, vocab_size) model predictions
        target_tokens: (B, L) ground truth tokens
        target_token_lens: (B,) actual sequence lengths
        tokenizer: Tokenizer with special tokens
        tolerance_window: Frame tolerance for matching
        
    Returns:
        Dictionary of metrics
    """
    # Get predicted tokens from logits
    # Shift by 1 to align predictions with targets (teacher forcing)
    pred_tokens = logits.argmax(dim=-1)  # (B, L-1)
    gt_tokens = target_tokens.contiguous()    # (B, L-1)
    
    # Adjust lengths
    adjusted_lens = target_token_lens
    
    # Compute metrics
    metrics_calculator = TurnTakingMetrics(tokenizer, tolerance_window)
    return metrics_calculator.compute_metrics(pred_tokens, gt_tokens, adjusted_lens)