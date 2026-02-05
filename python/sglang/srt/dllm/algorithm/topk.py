from typing import List, Tuple, Union
import time

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.dllm.algorithm.utils import AverageMeter


class TopK(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.k = config.algorithm_config.get("k", 1)
        # Track unmasked token counts for each forward pass
        self.transfer_token_counts = AverageMeter()
        self.forward_time = AverageMeter()
        self.num_forward_passes = 0

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        # Here, the forward_batch full logits contains all the blocks
        # such as [dllm_block_size * batch_size, hidden_size]
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id

        # Fast path: if there is no mask token, forward and save kv cache
        if torch.sum(mask_index).item() == 0:
            # Synchronize before timing to ensure clean measurement
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_start = time.perf_counter()
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            # Synchronize after forward to ensure GPU operations complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_time_ms = (time.perf_counter() - forward_start) * 1000.0
            self.forward_time.update(forward_time_ms)
            self.num_forward_passes += 1
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            next_token_ids = []
            return logits_output, next_token_ids, can_run_cuda_graph

        # Calculate start positions for each block
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)
            
            # Start trace recording for this block
            self.trace_recorder.start_block(
                batch_id=block_id,
                block_size=self.block_size,
                start_list=[start],
            )

        iteration = 0
        for _ in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            # Record forward pass time with CUDA synchronization for accurate timing
            # Synchronize before timing to ensure clean measurement
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_start = time.perf_counter()
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            # Synchronize after forward to ensure GPU operations complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_end = time.perf_counter()
            forward_time_ms = (forward_end - forward_start) * 1000.0
            self.forward_time.update(forward_time_ms)
            self.num_forward_passes += 1
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size
            
            iteration += 1
            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end,
                ]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    continue
                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end,
                ]

                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )
                x = torch.where(block_mask_index, x, block_input_ids)
                confidence = torch.where(block_mask_index, p, -np.inf)

                # TopK: statically unmask k tokens with highest confidence
                num_masked = block_mask_index.sum().item()
                k = min(self.k, num_masked)  # Don't select more than available masked tokens
                
                if k > 0:
                    _, select_indices = torch.topk(confidence, k=k)
                    transfer_index = torch.zeros_like(block_mask_index, dtype=torch.bool)
                    transfer_index[select_indices] = True
                else:
                    transfer_index = torch.zeros_like(block_mask_index, dtype=torch.bool)

                num_transfer_tokens = transfer_index.sum().item()
                self.transfer_token_counts.update(num_transfer_tokens)
                
                # Record trace information for this step
                selected_positions = torch.where(transfer_index)[0].cpu().tolist()
                selected_token_ids = x[transfer_index].cpu().tolist()
                confidence_values = confidence[transfer_index].cpu().tolist()
                
                self.trace_recorder.record_step(
                    iteration=iteration,
                    batch_id=batch_id,
                    block_start=curr_block_start,
                    block_end=curr_block_end,
                    num_masked_tokens=num_masked,
                    selected_token_ids=selected_token_ids,
                    selected_positions=selected_positions,
                    confidence_values=confidence_values,
                    num_transferred_tokens=num_transfer_tokens,
                    forward_time_ms=forward_time_ms,
                )

                block_input_ids[transfer_index] = x[transfer_index]

        
        # Record final forward pass time with CUDA synchronization for accurate timing
        # Synchronize before timing to ensure clean measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_start = time.perf_counter()
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        # Synchronize after forward to ensure GPU operations complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_end = time.perf_counter()
        forward_time_ms = (forward_end - forward_start) * 1000.0
        self.forward_time.update(forward_time_ms)
        self.num_forward_passes += 1
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        # Here next token ids is tricky to implement the dynamic lengths,
        # so we return a list of tensors
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]
        
        # End trace recording for each block
        for batch_id in range(batch_size):
            final_tokens = next_token_ids_list[batch_id].cpu().tolist()
            self.trace_recorder.end_block(
                batch_id=batch_id,
                total_iterations=iteration,
                total_forward_passes=self.num_forward_passes,
                final_token_ids=final_tokens,
            )

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = TopK
