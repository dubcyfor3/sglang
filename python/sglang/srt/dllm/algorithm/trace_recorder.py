"""
Trace recorder for DLLM decoding blocks.

This module provides functionality to record traces of decoding operations
for analysis and debugging purposes.
"""
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class DecodingStep:
    """Represents a single step in the decoding process."""
    iteration: int
    batch_id: int
    block_start: int
    block_end: int
    num_masked_tokens: int
    selected_token_ids: List[int]
    selected_positions: List[int]
    confidence_values: List[float]
    num_transferred_tokens: int
    forward_time_ms: float


@dataclass
class DecodingBlockTrace:
    """Represents a complete trace for a decoding block."""
    block_id: int
    batch_id: int
    block_size: int
    start_list: List[int]
    steps: List[DecodingStep]
    total_iterations: int
    total_forward_passes: int
    final_token_ids: List[int]


class TraceRecorder:
    """Records traces of DLLM decoding operations."""
    
    def __init__(self, enabled: bool = False, output_dir: Optional[str] = None):
        """
        Initialize the trace recorder.
        
        Args:
            enabled: Whether tracing is enabled
            output_dir: Directory to save trace files. If None, traces are only kept in memory.
        """
        self.enabled = enabled
        self.output_dir = output_dir
        self.traces: List[DecodingBlockTrace] = []
        # Support multiple concurrent blocks (one per batch_id)
        self.active_traces: Dict[int, DecodingBlockTrace] = {}
        self.active_steps: Dict[int, List[DecodingStep]] = {}
        self.block_counter = 0
        
        if self.enabled and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def start_block(
        self,
        batch_id: int,
        block_size: int,
        start_list: List[int],
    ):
        """Start recording a new decoding block."""
        if not self.enabled:
            return
        
        self.block_counter += 1
        self.active_steps[batch_id] = []
        self.active_traces[batch_id] = DecodingBlockTrace(
            block_id=self.block_counter,
            batch_id=batch_id,
            block_size=block_size,
            start_list=start_list,
            steps=[],
            total_iterations=0,
            total_forward_passes=0,
            final_token_ids=[],
        )
    
    def record_step(
        self,
        iteration: int,
        batch_id: int,
        block_start: int,
        block_end: int,
        num_masked_tokens: int,
        selected_token_ids: List[int],
        selected_positions: List[int],
        confidence_values: List[float],
        num_transferred_tokens: int,
        forward_time_ms: float,
    ):
        """Record a single decoding step."""
        if not self.enabled or batch_id not in self.active_traces:
            return
        
        step = DecodingStep(
            iteration=iteration,
            batch_id=batch_id,
            block_start=block_start,
            block_end=block_end,
            num_masked_tokens=num_masked_tokens,
            selected_token_ids=selected_token_ids,
            selected_positions=selected_positions,
            confidence_values=confidence_values,
            num_transferred_tokens=num_transferred_tokens,
            forward_time_ms=forward_time_ms,
        )
        if batch_id not in self.active_steps:
            self.active_steps[batch_id] = []
        self.active_steps[batch_id].append(step)
    
    def end_block(
        self,
        batch_id: int,
        total_iterations: int,
        total_forward_passes: int,
        final_token_ids: List[int],
    ):
        """End recording a block and save it."""
        if not self.enabled or batch_id not in self.active_traces:
            return
        
        trace = self.active_traces[batch_id]
        trace.steps = self.active_steps.get(batch_id, [])
        trace.total_iterations = total_iterations
        trace.total_forward_passes = total_forward_passes
        trace.final_token_ids = final_token_ids
        
        self.traces.append(trace)
        
        # Save to file if output directory is specified
        if self.output_dir:
            self._save_trace(trace)
        
        # Clean up
        del self.active_traces[batch_id]
        if batch_id in self.active_steps:
            del self.active_steps[batch_id]
    
    def _save_trace(self, trace: DecodingBlockTrace):
        """Save a trace to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"dllm_trace_block_{trace.block_id}_batch_{trace.batch_id}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to dict and handle numpy/torch types
        trace_dict = self._convert_to_serializable(asdict(trace))
        
        with open(filepath, 'w') as f:
            json.dump(trace_dict, f, indent=2)
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert numpy/torch types to native Python types."""
        import numpy as np
        import torch
        
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def get_all_traces(self) -> List[DecodingBlockTrace]:
        """Get all recorded traces."""
        return self.traces
    
    def clear(self):
        """Clear all recorded traces."""
        self.traces = []
        self.block_counter = 0
        self.active_traces = {}
        self.active_steps = {}
