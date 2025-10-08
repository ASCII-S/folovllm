"""Engine core for continuous batching (M2).

The EngineCore orchestrates the scheduler, executor, and sampler
to implement continuous batching with iteration-level scheduling.
"""

import time
from typing import Dict, List, Optional

from folovllm.config import ModelConfig, SchedulerConfig
from folovllm.core.sched import Scheduler
from folovllm.executor import GPUExecutor
from folovllm.sample import Sampler
from folovllm.engine.processor import InputProcessor
from folovllm.request import Request
from folovllm.outputs import RequestOutput
from folovllm.worker.input_batch import prepare_inputs_from_scheduler_output


class EngineCore:
    """Core engine for continuous batching.
    
    This class implements the main execution loop for M2:
    1. Schedule requests (which to process, how many tokens)
    2. Prepare batched inputs
    3. Execute model
    4. Sample next tokens
    5. Update scheduler state
    6. Return outputs
    
    For M2: Simple continuous batching
    For M3+: Will add PagedAttention, preemption, swapping
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        scheduler_config: SchedulerConfig,
        executor: GPUExecutor,
        sampler: Sampler,
        processor: InputProcessor,
    ):
        """Initialize EngineCore.
        
        Args:
            model_config: Model configuration
            scheduler_config: Scheduler configuration
            executor: GPU executor for model execution
            sampler: Sampler for token generation
            processor: Input processor for tokenization
        """
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.executor = executor
        self.sampler = sampler
        self.processor = processor
        
        # Create scheduler
        self.scheduler = Scheduler(
            model_config=model_config,
            scheduler_config=scheduler_config,
        )
        
        # Iteration counter
        self.iteration = 0
    
    def add_request(self, request: Request) -> None:
        """Add a request to the engine.
        
        Args:
            request: The request to add
        """
        self.scheduler.add_request(request)
    
    def has_unfinished_requests(self) -> bool:
        """Check if there are unfinished requests.
        
        Returns:
            True if there are requests waiting or running
        """
        return self.scheduler.has_unfinished_requests()
    
    def step(self) -> Dict[str, RequestOutput]:
        """Execute one iteration of the continuous batching loop.
        
        This is the core method that implements continuous batching:
        1. Scheduler decides which requests to process
        2. Prepare batched inputs
        3. Execute model
        4. Sample next tokens
        5. Update scheduler with results
        6. Return outputs for finished/updated requests
        
        Returns:
            Dict of req_id -> RequestOutput for requests with updates
        """
        self.iteration += 1
        
        # Step 1: Schedule requests
        scheduler_output = self.scheduler.schedule()
        
        # If nothing to schedule, return empty
        if scheduler_output.is_empty:
            return {}
        
        # Step 2: Prepare inputs
        input_batch = prepare_inputs_from_scheduler_output(scheduler_output)
        
        # Step 3: Execute model
        # Get logits for each request
        logits_dict = self.executor.execute_model_batch(input_batch)
        
        # Step 4: Sample next tokens
        # For each request, sample the next token from its logits
        sampled_tokens: Dict[str, int] = {}
        
        for req_id in input_batch.req_ids:
            if req_id not in logits_dict:
                continue
            
            logits = logits_dict[req_id]  # [vocab_size]
            
            # Get sampling params for this request
            request = self.scheduler.requests[req_id]
            sampling_params = request.sampling_params
            
            # Sample next token
            # Add batch dimension for sampler: [1, vocab_size]
            logits_batch = logits.unsqueeze(0)
            next_tokens, _ = self.sampler.sample(logits_batch, sampling_params)
            sampled_tokens[req_id] = next_tokens[0].item()
        
        # Step 5: Update scheduler with sampled tokens
        outputs = self.scheduler.update_from_output(
            scheduler_output,
            sampled_tokens,
        )
        
        # Step 6: Decode text for outputs
        for req_id, output in outputs.items():
            request = self.scheduler.requests[req_id]
            # Decode output tokens to text
            for completion in output.outputs:
                if completion.token_ids:
                    completion.text = self.processor.decode_tokens(
                        completion.token_ids,
                        skip_special_tokens=request.sampling_params.skip_special_tokens,
                    )
        
        # Step 7: Free caches for finished requests
        for req_id in scheduler_output.finished_req_ids:
            self.executor.free_request_cache(req_id)
        
        return outputs
    
    def get_request_counts(self) -> tuple[int, int]:
        """Get counts of requests in different states.
        
        Returns:
            Tuple of (num_running, num_waiting)
        """
        return self.scheduler.get_request_counts()
    
    def abort_requests(self, request_ids: List[str]) -> None:
        """Abort requests.
        
        Args:
            request_ids: List of request IDs to abort
        """
        from folovllm.request import RequestStatus
        self.scheduler.finish_requests(
            request_ids,
            RequestStatus.FINISHED_ABORTED,
        )
        
        # Free caches
        for req_id in request_ids:
            self.executor.free_request_cache(req_id)

