"""Top-k and Top-p sampling operations."""

import torch
import torch.nn.functional as F


def apply_top_k_filtering(
    logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Apply top-k filtering to logits.
    
    Keep only top k values, set others to -inf.
    
    Args:
        logits: Logits tensor [..., vocab_size]
        top_k: Number of top logits to keep
        
    Returns:
        Filtered logits [..., vocab_size]
    """
    if top_k <= 0:
        return logits
    
    # Get top-k values and indices
    top_k = min(top_k, logits.size(-1))
    top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
    
    # Create a mask for values not in top-k
    # Set all values to -inf first, then fill in top-k values
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(-1, top_k_indices, top_k_values)
    
    return filtered_logits


def apply_top_p_filtering(
    logits: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    """Apply top-p (nucleus) filtering to logits.
    
    Keep only tokens with cumulative probability >= top_p.
    
    Args:
        logits: Logits tensor [..., vocab_size]
        top_p: Cumulative probability threshold
        
    Returns:
        Filtered logits [..., vocab_size]
    """
    if top_p >= 1.0:
        return logits
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Remove tokens with cumulative probability above threshold
    # Keep at least one token
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift right to keep the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Create mask in original order
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    
    # Set removed indices to -inf
    filtered_logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    return filtered_logits


def apply_min_p_filtering(
    logits: torch.Tensor,
    min_p: float,
) -> torch.Tensor:
    """Apply min-p filtering to logits.
    
    Keep only tokens with probability >= min_p * max_prob.
    
    Args:
        logits: Logits tensor [..., vocab_size]
        min_p: Minimum probability threshold (relative to max)
        
    Returns:
        Filtered logits [..., vocab_size]
    """
    if min_p <= 0.0:
        return logits
    
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get max probability
    max_probs, _ = torch.max(probs, dim=-1, keepdim=True)
    
    # Create threshold
    threshold = min_p * max_probs
    
    # Mask out tokens below threshold
    mask = probs < threshold
    filtered_logits = logits.masked_fill(mask, float('-inf'))
    
    return filtered_logits

