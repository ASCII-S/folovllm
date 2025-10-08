"""Input batch for batched model execution."""

import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple

from folovllm.request import Request, Sequence


@dataclass
class InputBatch:
    """Batched input for model execution.
    
    Handles batching multiple sequences together with proper padding and masking.
    Supports both prefill (full prompts) and decode (single tokens) phases.
    
    For M2:
    - Separate prefill and decode batches (no mixing)
    - Simple padding to max length in batch
    - Causal attention mask
    
    M5+ will add:
    - Chunked prefill (mixed prefill/decode)
    - More efficient padding strategies
    """
    
    # List of requests in this batch
    requests: List[Request]
    
    # Token IDs tensor [batch_size, seq_len]
    token_ids: torch.Tensor
    
    # Position IDs tensor [batch_size, seq_len]
    position_ids: torch.Tensor
    
    # Attention mask [batch_size, seq_len, seq_len] or None (use causal)
    attention_mask: Optional[torch.Tensor] = None
    
    # Batch metadata
    batch_size: int = 0
    max_seq_len: int = 0
    is_prefill: bool = False
    
    # Sequence metadata (for output assembly)
    seq_ids: List[str] = None
    seq_lens: List[int] = None
    
    @classmethod
    def from_requests_prefill(
        cls,
        requests: List[Request],
        device: torch.device,
        pad_token_id: int = 0,
    ) -> "InputBatch":
        """Create a batch from requests in prefill phase.
        
        Prefill processes the entire prompt for each request.
        
        Args:
            requests: List of requests to batch.
            device: Device to place tensors on.
            pad_token_id: Token ID used for padding.
            
        Returns:
            InputBatch for prefill.
        """
        if not requests:
            raise ValueError("Cannot create batch from empty request list")
        
        # Collect sequences and their token IDs
        sequences = []
        token_lists = []
        
        for request in requests:
            # For M2, we only handle best_of=1 (single sequence per request)
            seqs = request.get_seqs()
            if not seqs:
                continue
            seq = seqs[0]
            sequences.append(seq)
            # In prefill, we use the full prompt
            token_lists.append(seq.data.prompt_token_ids)
        
        if not sequences:
            raise ValueError("No valid sequences in requests")
        
        # Get batch dimensions
        batch_size = len(sequences)
        max_seq_len = max(len(tokens) for tokens in token_lists)
        
        # Pad token lists to max length
        padded_tokens = []
        position_ids_list = []
        seq_lens = []
        
        for tokens in token_lists:
            seq_len = len(tokens)
            seq_lens.append(seq_len)
            
            # Pad tokens
            padded = tokens + [pad_token_id] * (max_seq_len - seq_len)
            padded_tokens.append(padded)
            
            # Create position IDs [0, 1, 2, ..., seq_len-1, 0, 0, ...]
            positions = list(range(seq_len)) + [0] * (max_seq_len - seq_len)
            position_ids_list.append(positions)
        
        # Convert to tensors
        token_ids = torch.tensor(padded_tokens, dtype=torch.long, device=device)
        position_ids = torch.tensor(position_ids_list, dtype=torch.long, device=device)
        
        # Create attention mask for padding
        # Shape: [batch_size, max_seq_len]
        # 1 for real tokens, 0 for padding
        attention_mask = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.bool,
            device=device,
        )
        for i, seq_len in enumerate(seq_lens):
            attention_mask[i, :seq_len] = True
        
        # Convert to 4D attention mask for compatibility
        # Shape: [batch_size, 1, max_seq_len, max_seq_len]
        # Causal mask + padding mask
        causal_mask = torch.triu(
            torch.ones((max_seq_len, max_seq_len), dtype=torch.bool, device=device),
            diagonal=1,
        )
        # Invert: 1 where we can attend, 0 where we can't
        causal_mask = ~causal_mask
        
        # Combine with padding mask
        attention_mask_4d = attention_mask.unsqueeze(1).unsqueeze(2) & causal_mask.unsqueeze(0).unsqueeze(0)
        
        return cls(
            requests=requests,
            token_ids=token_ids,
            position_ids=position_ids,
            attention_mask=attention_mask_4d,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            is_prefill=True,
            seq_ids=[seq.seq_id for seq in sequences],
            seq_lens=seq_lens,
        )
    
    @classmethod
    def from_requests_decode(
        cls,
        requests: List[Request],
        device: torch.device,
    ) -> "InputBatch":
        """Create a batch from requests in decode phase.
        
        Decode generates one token at a time for each sequence.
        
        Args:
            requests: List of requests to batch.
            device: Device to place tensors on.
            
        Returns:
            InputBatch for decode.
        """
        if not requests:
            raise ValueError("Cannot create batch from empty request list")
        
        # Collect sequences and their last tokens
        sequences = []
        last_tokens = []
        seq_lens = []
        
        for request in requests:
            seqs = request.get_seqs()
            if not seqs:
                continue
            seq = seqs[0]
            sequences.append(seq)
            
            # In decode, we only use the last generated token
            last_token_id = seq.get_last_token_id()
            last_tokens.append(last_token_id)
            
            # Current sequence length (prompt + generated tokens)
            seq_lens.append(seq.get_len())
        
        if not sequences:
            raise ValueError("No valid sequences in requests")
        
        batch_size = len(sequences)
        
        # Token IDs: [batch_size, 1] - just the last token
        token_ids = torch.tensor(
            [[token] for token in last_tokens],
            dtype=torch.long,
            device=device,
        )
        
        # Position IDs: [batch_size, 1] - position of next token
        position_ids = torch.tensor(
            [[seq_len] for seq_len in seq_lens],
            dtype=torch.long,
            device=device,
        )
        
        # For decode, attention mask is handled by KV cache
        # We don't need explicit attention mask here
        
        return cls(
            requests=requests,
            token_ids=token_ids,
            position_ids=position_ids,
            attention_mask=None,  # Will use KV cache mask
            batch_size=batch_size,
            max_seq_len=1,  # Only 1 new token
            is_prefill=False,
            seq_ids=[seq.seq_id for seq in sequences],
            seq_lens=seq_lens,
        )
    
    def to(self, device: torch.device) -> "InputBatch":
        """Move batch to a different device.
        
        Args:
            device: Target device.
            
        Returns:
            New InputBatch on the target device.
        """
        return InputBatch(
            requests=self.requests,
            token_ids=self.token_ids.to(device),
            position_ids=self.position_ids.to(device),
            attention_mask=self.attention_mask.to(device) if self.attention_mask is not None else None,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            is_prefill=self.is_prefill,
            seq_ids=self.seq_ids,
            seq_lens=self.seq_lens,
        )
    
    def __repr__(self) -> str:
        return (
            f"InputBatch("
            f"batch_size={self.batch_size}, "
            f"max_seq_len={self.max_seq_len}, "
            f"is_prefill={self.is_prefill})"
        )

