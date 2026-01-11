import sglang as sgl
import argparse
import json
import time
import random
import numpy as np
from transformers import AutoTokenizer
from utils import load_config, save_config, set_block_size_and_threshold
from dataset_utils import (
    load_sharegpt_dataset,
    load_gsm8k_dataset,
    load_humaneval_dataset,
    evaluate_gsm8k,
    RANDOM_SEED,
)

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def get_dllm_stats(llm):
    """Extract DLLM statistics from engine."""
    server_info = llm.get_server_info()
    internal_states = server_info.get("internal_states", [])
    
    if internal_states and isinstance(internal_states, list) and len(internal_states) > 0:
        state = internal_states[0]
        result = {}
        
        if "dllm_transfer_token_counts" in state:
            dllm_stats = state["dllm_transfer_token_counts"]
            result["transfer_token_counts"] = {
                "avg": dllm_stats.get('avg', 0),
                "count": dllm_stats.get('count', 0),
                "sum": dllm_stats.get('sum', 0),
            }
        
        if "dllm_num_forward_passes" in state:
            result["num_forward_passes"] = state.get("dllm_num_forward_passes", 0)
        
        return result if result else None
    return None

def test_gsm8k(args):
    """
    Test GSM8K accuracy following the unittest pattern.
    Similar to test_llada2_mini.py::TestLLaDA2Mini.test_gsm8k but using direct Engine API.
    
    Returns:
        dict: Metrics containing accuracy, invalid, latency, and output_throughput
    """
    # Set DLLM configuration
    set_block_size_and_threshold(args.dllm_algorithm_config, args.block_size, args.confidence_threshold)
    
    # Load tokenizer for dataset processing
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Create dataset args
    class DatasetArgs:
        num_shots = getattr(args, 'num_shots', 5)
    dataset_args = DatasetArgs()
    
    # Load GSM8K dataset
    print(f"Loading GSM8K dataset with {args.num_questions} questions...")
    result = load_gsm8k_dataset(dataset_args, args.num_questions, tokenizer)
    prompts, (labels, output_lens) = result
    
    print(f"Loaded {len(prompts)} prompts")
    if output_lens:
        print(f"Output lengths: min={min(output_lens)}, max={max(output_lens)}, avg={sum(output_lens)/len(output_lens):.1f}")
    
    # Initialize engine with DLLM
    print("Initializing SGLang engine with DLLM...")
    llm = sgl.Engine(
        model_path=args.model_path,
        dllm_algorithm=args.dllm_algorithm,
        dllm_algorithm_config=args.dllm_algorithm_config,
        max_running_requests=getattr(args, 'max_running_requests', 1),
        mem_fraction_static=getattr(args, 'mem_fraction_static', None),
        trust_remote_code=True,
        log_level="error",  # Reduce verbosity for testing
    )
    
    # Prepare sampling parameters
    # For accuracy tests: use EOS and stop strings for natural stopping behavior
    # Use per-request output lengths if available, otherwise use default
    if output_lens and len(output_lens) == len(prompts) and all(olen is not None for olen in output_lens):
        sampling_params_list = []
        for output_len in output_lens:
            params = {
                "temperature": getattr(args, 'temperature', 0),
                "max_new_tokens": min(output_len + 50, getattr(args, 'max_new_tokens', 512)),  # Add buffer
                "ignore_eos": False,  # Use EOS tokens for natural stopping (accuracy test)
            }
            params["stop"] = ["Question", "Assistant:", "<|separator|>"]  # Stop strings for accuracy
            sampling_params_list.append(params)
        sampling_params = sampling_params_list
        print(f"Using per-request output lengths (dataset-specific)")
        print(f"Accuracy test mode: ignore_eos=False, stop strings enabled")
    else:
        sampling_params = {
            "temperature": getattr(args, 'temperature', 0),
            "max_new_tokens": getattr(args, 'max_new_tokens', 512),
            "ignore_eos": False,  # Use EOS tokens for natural stopping (accuracy test)
            "stop": ["Question", "Assistant:", "<|separator|>"],  # Stop strings for accuracy
        }
        print(f"Using default max_new_tokens: {sampling_params['max_new_tokens']}")
        print(f"Accuracy test mode: ignore_eos=False, stop strings enabled")
    
    # Run inference
    print(f"Running inference on {len(prompts)} prompts...")
    tic = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    latency = time.perf_counter() - tic
    
    print(f"Inference completed in {latency:.3f} seconds")
    
    # Normalize outputs to list format
    if not isinstance(outputs, list):
        outputs = [outputs]
    
    # Extract text from outputs
    output_texts = []
    for output in outputs:
        if isinstance(output, dict):
            output_texts.append(output.get("text", ""))
        else:
            output_texts.append(str(output))
    
    # Evaluate accuracy
    eval_results = evaluate_gsm8k(output_texts, labels)
    accuracy = eval_results["accuracy"]
    invalid = eval_results["invalid_rate"]
    
    # Compute throughput
    total_tokens = 0
    for output in outputs:
        if isinstance(output, dict):
            meta_info = output.get("meta_info", {})
            if isinstance(meta_info, dict):
                total_tokens += meta_info.get("completion_tokens", 0)
    
    output_throughput = total_tokens / latency if latency > 0 else 0.0
    
    # Get DLLM stats if available
    dllm_stats = get_dllm_stats(llm)
    
    # Print results (matching unittest format)
    print(f"\n{'='*60}")
    print(f"GSM8K Evaluation Results")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")
    print(f"Total output tokens: {total_tokens}")
    print(f"Num correct: {eval_results['num_correct']}/{eval_results['num_total']}")
    
    if dllm_stats:
        print(f"\nDLLM Stats:")
        if "transfer_token_counts" in dllm_stats:
            transfer_stats = dllm_stats["transfer_token_counts"]
            print(f"  Transfer token counts - avg: {transfer_stats['avg']:.2f}")
        if "num_forward_passes" in dllm_stats:
            print(f"  Number of forward passes: {dllm_stats['num_forward_passes']}")
    
    # Cleanup
    llm.shutdown()
    
    # Return metrics in the same format as unittest test
    metrics = {
        "accuracy": accuracy,
        "invalid": invalid,
        "latency": latency,
        "output_throughput": output_throughput,
        "dllm_stats": dllm_stats,
    }
    
    return metrics

def main(args):
    set_block_size_and_threshold(args.dllm_algorithm_config, args.block_size, args.confidence_threshold)
    
    # Load tokenizer for dataset processing
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Create a simple args object with defaults for dataset loading
    class DatasetArgs:
        pass
    dataset_args = DatasetArgs()
    
    # Load dataset based on type
    print(f"Loading {args.dataset} dataset...")
    output_lens = []
    if args.dataset == "sharegpt":
        prompts, dataset_info = load_sharegpt_dataset(dataset_args, tokenizer, args.num_samples)
        # Extract output lengths from ShareGPT requests
        output_lens = [req.output_len for req in dataset_info]
    elif args.dataset == "gsm8k":
        result = load_gsm8k_dataset(dataset_args, args.num_samples, tokenizer)
        prompts, (labels, output_lens) = result
    elif args.dataset == "humaneval":
        result = load_humaneval_dataset(dataset_args, args.num_samples, tokenizer)
        prompts, (labels, output_lens) = result
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Choose from: sharegpt, gsm8k, humaneval")
    
    print(f"Loaded {len(prompts)} samples")
    if output_lens:
        print(f"Output lengths: min={min(output_lens)}, max={max(output_lens)}, avg={sum(output_lens)/len(output_lens):.1f}")
    
    # Initialize engine
    llm = sgl.Engine(
        model_path=args.model_path,
        dllm_algorithm=args.dllm_algorithm,
        dllm_algorithm_config=args.dllm_algorithm_config,
        max_running_requests=args.max_running_requests,
        mem_fraction_static=getattr(args, 'mem_fraction_static', None),
        trust_remote_code=True
    )

    # Prepare sampling parameters
    # For performance benchmarks: ignore EOS and stop strings to get consistent token counts
    # Use dataset-specific output lengths if available, otherwise use default
    if output_lens and len(output_lens) == len(prompts) and all(olen is not None for olen in output_lens):
        # Create per-request sampling parameters with dataset-specific output lengths
        sampling_params_list = []
        for output_len in output_lens:
            params = {
                "temperature": getattr(args, 'temperature', 0),
                "max_new_tokens": output_len,
                "ignore_eos": True,  # Ignore EOS for consistent benchmark measurements
            }
            # Don't add stop strings for benchmarks - we want exact token counts
            # Stop strings are only used in accuracy tests (test_gsm8k)
            sampling_params_list.append(params)
        sampling_params = sampling_params_list
        print(f"Using per-request output lengths (dataset-specific)")
        print(f"Benchmark mode: ignore_eos=True, no stop strings")
    else:
        # Fallback to single sampling params for all requests
        sampling_params = {
            "temperature": getattr(args, 'temperature', 0),
            "max_new_tokens": getattr(args, 'max_new_tokens', 1024),
            "ignore_eos": True,  # Ignore EOS for consistent benchmark measurements
        }
        # Don't add stop strings for benchmarks
        print(f"Using default max_new_tokens: {sampling_params['max_new_tokens']}")
        print(f"Benchmark mode: ignore_eos=True, no stop strings")

    # Run inference
    print(f"Running inference on {len(prompts)} prompts...")
    tic = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    latency = time.perf_counter() - tic
    
    print(f"Inference completed in {latency:.3f} seconds")
    
    # Normalize outputs to list format
    if not isinstance(outputs, list):
        outputs = [outputs]
    
    print(f"\nCompleted {len(outputs)} generations")
    
    # Get DLLM stats
    dllm_stats = get_dllm_stats(llm)
    if dllm_stats:
        print(f"\nDLLM Stats:")
        if "transfer_token_counts" in dllm_stats:
            transfer_stats = dllm_stats["transfer_token_counts"]
            print(f"  Transfer token counts - avg: {transfer_stats['avg']:.2f}")
        if "num_forward_passes" in dllm_stats:
            print(f"  Number of forward passes: {dllm_stats['num_forward_passes']}")
    else:
        print("\nDLLM stats not available")
    
    # Compute throughput
    total_tokens = 0
    for output in outputs:
        if isinstance(output, dict):
            meta_info = output.get("meta_info", {})
            if isinstance(meta_info, dict):
                total_tokens += meta_info.get("completion_tokens", 0)
    
    if total_tokens > 0:
        throughput = total_tokens / latency
        print(f"\nThroughput: {throughput:.2f} tokens/second")
        print(f"Total output tokens: {total_tokens}")
    
    # Save results if requested
    if hasattr(args, 'output_file') and args.output_file:
        results = {
            "dataset": args.dataset,
            "num_samples": len(prompts),
            "latency": latency,
            "dllm_stats": dllm_stats,
            "throughput": total_tokens / latency if total_tokens > 0 else 0,
        }
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate DLLM on various datasets")
    
    # Model and DLLM arguments
    parser.add_argument("--model_path", type=str, default="inclusionAI/LLaDA2.0-mini")
    parser.add_argument("--dllm_algorithm", type=str, default="LowConfidence")
    parser.add_argument("--dllm_algorithm_config", type=str, default="workspace/config.yaml")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--confidence_threshold", type=float, default=0.90)
    parser.add_argument("--max_running_requests", type=int, default=16)
    parser.add_argument("--mem_fraction_static", type=float, default=None,
                       help="Fraction of GPU memory for static allocation (model weights + KV cache). Default is auto-calculated (~0.9). Use smaller values (e.g., 0.7, 0.8) to reduce GPU utilization.")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="sharegpt", 
                       choices=["sharegpt", "gsm8k", "humaneval"],
                       help="Dataset to evaluate on")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--num_questions", type=int, default=200,
                       help="Number of GSM8K questions (for test_gsm8k)")
    parser.add_argument("--num_shots", type=int, default=5,
                       help="Number of few-shot examples for GSM8K")
    parser.add_argument("--test_gsm8k", action="store_true",
                       help="Run GSM8K test following unittest pattern")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    
    # Output
    parser.add_argument("--output_file", type=str, default=None,
                       help="Path to save evaluation results JSON")
    
    args = parser.parse_args()
    
    if args.test_gsm8k:
        # Run GSM8K test following unittest pattern
        metrics = test_gsm8k(args)
        print(f"\n{metrics=}")
        
        # Optionally assert thresholds (like unittest)
        if hasattr(args, 'min_accuracy') and args.min_accuracy:
            assert metrics["accuracy"] >= args.min_accuracy, \
                f"Accuracy {metrics['accuracy']:.3f} below threshold {args.min_accuracy}"
        if hasattr(args, 'min_throughput') and args.min_throughput:
            assert metrics["output_throughput"] >= args.min_throughput, \
                f"Throughput {metrics['output_throughput']:.3f} below threshold {args.min_throughput}"
    else:
        main(args)