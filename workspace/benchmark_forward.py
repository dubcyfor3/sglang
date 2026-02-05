import sglang as sgl
import argparse
import time
import torch
import numpy as np
import yaml
from transformers import AutoTokenizer


def set_algorithm_config(config_path, block_size, k=None, threshold=None):
    """
    Set the algorithm configuration in a YAML config file.
    
    Args:
        config_path: Path to the YAML config file
        block_size: The block size value to set
        k: The k value for TopK algorithm (optional)
        threshold: The confidence threshold for LowConfidence algorithm (optional)
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {}
    
    config['block_size'] = block_size
    if k is not None:
        config['k'] = k
    if threshold is not None:
        config['threshold'] = threshold
    
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    print(f"Config updated: block_size={block_size}, k={k}, threshold={threshold}")
    print(f"Config saved to {config_path}")


def create_prompt_of_length(tokenizer, target_length: int, base_text: str = "The quick brown fox jumps over the lazy dog. "):
    """
    Create a prompt that tokenizes to exactly target_length tokens.
    
    Args:
        tokenizer: The tokenizer to use
        target_length: Desired number of tokens
        base_text: Base text to repeat/extend
        
    Returns:
        str: Prompt text that tokenizes to target_length tokens
    """
    # Start with base text
    prompt = base_text
    current_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    current_length = len(current_ids)
    
    if current_length == target_length:
        return prompt
    
    if current_length < target_length:
        # Need to add more tokens
        # Add words one by one until we reach target
        words = base_text.split()
        word_idx = 0
        while current_length < target_length:
            # Add next word
            if word_idx >= len(words):
                word_idx = 0  # Repeat if needed
            prompt += words[word_idx] + " "
            word_idx += 1
            current_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
            current_length = len(current_ids)
            
            # Safety check to avoid infinite loop
            if current_length > target_length * 2:
                break
        
        # If we overshot, trim by removing words
        while current_length > target_length:
            # Remove last word
            words_list = prompt.split()
            if len(words_list) > 1:
                prompt = " ".join(words_list[:-1]) + " "
            else:
                break
            current_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
            current_length = len(current_ids)
    
    else:
        # Need to remove tokens
        words = prompt.split()
        while current_length > target_length and len(words) > 1:
            words = words[:-1]
            prompt = " ".join(words) + " "
            current_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
            current_length = len(current_ids)
    
    # Final verification
    final_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    final_length = len(final_ids)
    
    if final_length != target_length:
        print(f"Warning: Could not create exact prompt length. Got {final_length}, wanted {target_length}")
    
    return prompt


def benchmark_forward_pass(
    model_path: str,
    dllm_algorithm: str = "TopK",
    dllm_algorithm_config: str = "workspace/config.yaml",
    block_size: int = 32,
    k: int = 1,
    confidence_threshold: float = 0.90,
    batch_size: int = 1,
    input_length: int = 256,
    output_length: int = 256,
    num_warmup: int = 3,
    num_iterations: int = 10,
    disable_cuda_graph: bool = False,
):
    """
    Benchmark forward pass time for DLLM.
    
    Args:
        model_path: Path to the model
        dllm_algorithm: DLLM algorithm name (TopK or LowConfidence)
        dllm_algorithm_config: Path to DLLM config file
        block_size: DLLM block size
        k: TopK algorithm parameter (number of tokens to unmask per iteration)
        confidence_threshold: LowConfidence algorithm threshold
        batch_size: Number of requests in batch
        input_length: Input prompt length in tokens (default: 256)
        output_length: Output length in tokens (default: 256)
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        disable_cuda_graph: Whether to disable CUDA graph
    """
    # Load tokenizer to create dummy input
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Create prompts of exactly input_length tokens
    print(f"Creating prompts of length {input_length} tokens...")
    base_prompt = create_prompt_of_length(tokenizer, input_length)
    
    # Verify prompt length
    base_ids = tokenizer.encode(base_prompt, return_tensors="pt")[0]
    actual_length = len(base_ids)
    print(f"Created prompt with {actual_length} tokens (target: {input_length})")
    
    # Create prompts for batch (all same length)
    prompts = [base_prompt] * batch_size
    
    print(f"\nInitializing SGLang engine with DLLM...")
    print(f"  Model: {model_path}")
    print(f"  DLLM Algorithm: {dllm_algorithm}")
    print(f"  Block Size: {block_size}")
    if dllm_algorithm == "TopK":
        print(f"  TopK k: {k}")
    else:
        print(f"  Confidence Threshold: {confidence_threshold}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Input Length: {input_length} tokens")
    print(f"  Output Length: {output_length} tokens")
    print(f"  Disable CUDA Graph: {disable_cuda_graph}")
    
    # Initialize engine
    llm = sgl.Engine(
        model_path=model_path,
        dllm_algorithm=dllm_algorithm,
        dllm_algorithm_config=dllm_algorithm_config,
        max_running_requests=batch_size,
        trust_remote_code=True,
        disable_cuda_graph=disable_cuda_graph,
        log_level="error",
    )
    
    # Prepare sampling parameters
    # For DLLM, we want to generate exactly output_length tokens
    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": output_length,
        "ignore_eos": True,  # Generate exactly output_length tokens
    }
    
    print(f"\nWarming up ({num_warmup} iterations)...")
    for i in range(num_warmup):
        _ = llm.generate(prompts, sampling_params)
    
    # Synchronize GPU before benchmarking
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print(f"\nBenchmarking forward pass ({num_iterations} iterations)...")
    forward_times = []
    
    for i in range(num_iterations):
        # Synchronize before each iteration
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure forward pass time
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.perf_counter()
        
        forward_time = end_time - start_time
        forward_times.append(forward_time)
        
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/{num_iterations}: {forward_time*1000:.2f} ms")
    
    # Calculate statistics
    forward_times = np.array(forward_times)
    avg_time = np.mean(forward_times)
    min_time = np.min(forward_times)
    max_time = np.max(forward_times)
    std_time = np.std(forward_times)
    median_time = np.median(forward_times)
    
    # Calculate throughput
    total_output_tokens = batch_size * output_length
    avg_throughput = total_output_tokens / avg_time if avg_time > 0 else 0
    
    # Get DLLM stats if available
    dllm_stats = None
    try:
        server_info = llm.get_server_info()
        internal_states = server_info.get("internal_states", [])
        if internal_states and len(internal_states) > 0:
            state = internal_states[0]
            dllm_stats = {
                "num_forward_passes": state.get("dllm_num_forward_passes", 0),
                "transfer_token_counts": state.get("dllm_transfer_token_counts", {}),
                "forward_time": state.get("dllm_forward_time", {}),
            }
    except:
        pass
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Forward Pass Benchmark Results")
    print(f"{'='*60}")
    print(f"Average time:     {avg_time*1000:.2f} ms")
    print(f"Median time:      {median_time*1000:.2f} ms")
    print(f"Min time:         {min_time*1000:.2f} ms")
    print(f"Max time:         {max_time*1000:.2f} ms")
    print(f"Std deviation:    {std_time*1000:.2f} ms")
    print(f"\nThroughput:       {avg_throughput:.2f} tokens/second")
    print(f"Total output tokens: {total_output_tokens} tokens/batch")
    print(f"Input tokens:     {batch_size * input_length} tokens/batch")
    
    if dllm_stats:
        print(f"\nDLLM Stats:")
        if "num_forward_passes" in dllm_stats:
            print(f"  Forward passes:  {dllm_stats['num_forward_passes']}")
        if "transfer_token_counts" in dllm_stats:
            transfer = dllm_stats["transfer_token_counts"]
            if isinstance(transfer, dict) and "avg" in transfer:
                print(f"  Transfer tokens (avg): {transfer['avg']:.2f}")
        if "forward_time" in dllm_stats:
            forward_time = dllm_stats["forward_time"]
            if isinstance(forward_time, dict) and "avg" in forward_time:
                print(f"  Forward time (avg): {forward_time['avg']:.2f} ms")
    
    print(f"{'='*60}\n")
    
    # Cleanup
    llm.shutdown()
    
    return {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": std_time,
        "median_time": median_time,
        "throughput": avg_throughput,
        "dllm_stats": dllm_stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DLLM forward pass time")
    
    # Model and DLLM arguments
    parser.add_argument("--model_path", type=str, default="inclusionAI/LLaDA2.0-mini",
                       help="Path to the model")
    parser.add_argument("--dllm_algorithm", type=str, default="TopK",
                       help="DLLM algorithm name (TopK or LowConfidence)")
    parser.add_argument("--dllm_algorithm_config", type=str, default="workspace/config.yaml",
                       help="Path to DLLM algorithm config file")
    parser.add_argument("--block_size", type=int, default=32,
                       help="DLLM block size")
    parser.add_argument("--k", type=int, default=1,
                       help="TopK algorithm: number of tokens to unmask per iteration")
    parser.add_argument("--confidence_threshold", type=float, default=0.90,
                       help="LowConfidence algorithm: confidence threshold")
    
    # Benchmark arguments
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (number of requests)")
    parser.add_argument("--input_length", type=int, default=256,
                       help="Input prompt length in tokens")
    parser.add_argument("--output_length", type=int, default=256,
                       help="Output length in tokens")
    parser.add_argument("--num_warmup", type=int, default=3,
                       help="Number of warmup iterations")
    parser.add_argument("--num_iterations", type=int, default=10,
                       help="Number of benchmark iterations")
    parser.add_argument("--disable_cuda_graph", action="store_true", default=False,
                       help="Disable CUDA graph")
    
    args = parser.parse_args()
    
    # Update config file with algorithm-specific parameters
    if args.dllm_algorithm == "TopK":
        set_algorithm_config(
            args.dllm_algorithm_config,
            args.block_size,
            k=args.k,
            threshold=None
        )
    else:
        set_algorithm_config(
            args.dllm_algorithm_config,
            args.block_size,
            k=None,
            threshold=args.confidence_threshold
        )
    
    benchmark_forward_pass(
        model_path=args.model_path,
        dllm_algorithm=args.dllm_algorithm,
        dllm_algorithm_config=args.dllm_algorithm_config,
        block_size=args.block_size,
        k=args.k,
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size,
        input_length=args.input_length,
        output_length=args.output_length,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        disable_cuda_graph=args.disable_cuda_graph,
    )
