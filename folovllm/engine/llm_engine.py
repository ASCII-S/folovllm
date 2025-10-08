"""LLM Engine for text generation."""

import time
import torch
from typing import Iterator, Optional, Union

from folovllm.config import ModelConfig
from folovllm.model_loader import ModelLoader
from folovllm.executor import GPUExecutor
from folovllm.engine.processor import InputProcessor
from folovllm.sample import Sampler
from folovllm.sampling_params import SamplingParams
from folovllm.request import Request, SequenceStatus
from folovllm.outputs import RequestOutput, CompletionOutput


class LLMEngine:
    """Main LLM engine for text generation.
    
    This is the primary user-facing interface for M1.
    Provides synchronous generation for single requests.
    
    M2 will add:
    - Asynchronous generation
    - Continuous batching
    - Multi-request handling
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        device: Optional[str] = None,
    ):
        """Initialize LLM engine.
        
        Args:
            model_config: Model configuration
            device: Device to run on (e.g., 'cuda:0')
        """
        self.model_config = model_config
        
        # Load tokenizer
        print("Loading tokenizer...")
        loader = ModelLoader(model_config)
        self.tokenizer = loader.load_tokenizer()
        
        # Create executor
        print("Initializing executor...")
        self.executor = GPUExecutor(model_config, device)
        
        # Create processor and sampler
        self.processor = InputProcessor(self.tokenizer)
        self.sampler = Sampler()
        
        print("LLM Engine initialized successfully!")
    
    def generate(
        self,
        prompt: Union[str, Request],
        sampling_params: Optional[SamplingParams] = None,
        return_outputs: bool = True,
    ) -> Union[RequestOutput, Iterator[RequestOutput]]:
        """Generate text for a single prompt.
        
        For M1: synchronous generation only.
        For M2+: will support streaming.
        
        Args:
            prompt: Input prompt text or Request object
            sampling_params: Sampling parameters (required if prompt is str)
            return_outputs: Whether to return full outputs (vs streaming)
            
        Returns:
            RequestOutput object with generated text
        """
        # Process request
        if isinstance(prompt, str):
            if sampling_params is None:
                sampling_params = SamplingParams()
            request = self.processor.process_request(prompt, sampling_params)
        else:
            request = prompt
            sampling_params = request.sampling_params
        
        # Clear KV caches from previous generation
        self.executor.clear_kv_caches()
        
        # Run generation loop
        output = self._generate_single(request)
        
        return output
    
    def _generate_single(self, request: Request) -> RequestOutput:
        """Generate text for a single request.
        
        Args:
            request: Request object
            
        Returns:
            RequestOutput with generated text
        """
        sampling_params = request.sampling_params
        
        # Get first sequence (for M1, we only handle n=1)
        sequences = request.get_seqs()
        if len(sequences) > 1:
            raise NotImplementedError(
                "M1 only supports n=1. Multiple sequences will be supported in M2."
            )
        seq = sequences[0]
        
        # Mark sequence as running
        seq.status = SequenceStatus.RUNNING
        
        # Get prompt tokens
        prompt_token_ids = seq.data.prompt_token_ids
        prompt_len = len(prompt_token_ids)
        
        # Prefill phase: process entire prompt
        start_time = time.time()
        input_tokens = torch.tensor([prompt_token_ids], dtype=torch.long)
        logits = self.executor.get_next_token_logits(input_tokens, start_pos=0)
        
        # Sample first token
        next_tokens, _ = self.sampler.sample(logits, sampling_params)
        next_token_id = next_tokens[0].item()
        seq.add_token_id(next_token_id)
        
        first_token_time = time.time()
        ttft = first_token_time - start_time
        
        # Decode phase: generate tokens one by one
        decode_times = []
        for step in range(1, sampling_params.max_tokens or 100):
            decode_start = time.time()
            
            # Check stop conditions
            output_text = self.processor.decode_tokens(
                seq.data.output_token_ids,
                skip_special_tokens=sampling_params.skip_special_tokens,
            )
            should_stop, finish_reason = self.sampler.check_stop_conditions(
                seq.data.output_token_ids,
                output_text,
                sampling_params,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            if should_stop:
                seq.status = SequenceStatus.FINISHED_STOPPED if finish_reason == "stop" \
                    else SequenceStatus.FINISHED_LENGTH_CAPPED
                break
            
            # Generate next token
            input_tokens = torch.tensor([[next_token_id]], dtype=torch.long)
            logits = self.executor.get_next_token_logits(
                input_tokens,
                start_pos=prompt_len + step,
            )
            
            # Sample next token
            next_tokens, _ = self.sampler.sample(logits, sampling_params)
            next_token_id = next_tokens[0].item()
            seq.add_token_id(next_token_id)
            
            decode_times.append(time.time() - decode_start)
        
        # Build output
        output = self._build_output(request)
        
        # Add timing metrics
        total_time = time.time() - start_time
        num_tokens = len(seq.data.output_token_ids)
        tpot = sum(decode_times) / len(decode_times) if decode_times else 0
        
        output.metrics = {
            "ttft": ttft,  # Time to first token
            "tpot": tpot,  # Time per output token
            "total_time": total_time,
            "throughput": num_tokens / total_time if total_time > 0 else 0,
        }
        
        return output
    
    def _build_output(self, request: Request) -> RequestOutput:
        """Build RequestOutput from Request.
        
        Args:
            request: Request object
            
        Returns:
            RequestOutput
        """
        sequences = request.get_seqs()
        
        # Build completion outputs
        completion_outputs = []
        for idx, seq in enumerate(sequences):
            # Decode output tokens
            output_text = self.processor.decode_tokens(
                seq.data.output_token_ids,
                skip_special_tokens=request.sampling_params.skip_special_tokens,
            )
            
            # Determine finish reason
            finish_reason = None
            if seq.status == SequenceStatus.FINISHED_STOPPED:
                finish_reason = "stop"
            elif seq.status == SequenceStatus.FINISHED_LENGTH_CAPPED:
                finish_reason = "length"
            
            completion_output = CompletionOutput(
                index=idx,
                text=output_text,
                token_ids=seq.data.output_token_ids.copy(),
                cumulative_logprob=None,  # Not implemented in M1
                logprobs=None,
                finish_reason=finish_reason,
            )
            completion_outputs.append(completion_output)
        
        # Check if all sequences are finished
        finished = all(seq.is_finished() for seq in sequences)
        
        output = RequestOutput(
            request_id=request.request_id,
            prompt=request.prompt,
            prompt_token_ids=request.prompt_token_ids,
            outputs=completion_outputs,
            finished=finished,
        )
        
        return output

