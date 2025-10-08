"""Input processor for tokenization and request creation."""

import uuid
from typing import List, Union
from transformers import PreTrainedTokenizer

from folovllm.request import Request
from folovllm.sampling_params import SamplingParams


class InputProcessor:
    """Processes input prompts into Request objects.
    
    Handles:
    - Tokenization
    - Request object creation
    - Input validation
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """Initialize input processor.
        
        Args:
            tokenizer: HuggingFace tokenizer
        """
        self.tokenizer = tokenizer
    
    def process_request(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str = None,
    ) -> Request:
        """Process a single request.
        
        Args:
            prompt: Input prompt text
            sampling_params: Sampling parameters
            request_id: Optional request ID (auto-generated if None)
            
        Returns:
            Request object
        """
        # Generate request ID if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # Tokenize prompt
        prompt_token_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
        )
        
        # Create request
        request = Request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
        
        return request
    
    def process_requests(
        self,
        prompts: List[str],
        sampling_params: Union[SamplingParams, List[SamplingParams]],
    ) -> List[Request]:
        """Process multiple requests.
        
        Args:
            prompts: List of prompt texts
            sampling_params: Single SamplingParams or list of SamplingParams
            
        Returns:
            List of Request objects
        """
        # Handle single sampling params for all requests
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        
        # Validate lengths match
        if len(prompts) != len(sampling_params):
            raise ValueError(
                f"Number of prompts ({len(prompts)}) must match "
                f"number of sampling_params ({len(sampling_params)})"
            )
        
        # Process each request
        requests = []
        for prompt, params in zip(prompts, sampling_params):
            request = self.process_request(prompt, params)
            requests.append(request)
        
        return requests
    
    def decode_tokens(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

