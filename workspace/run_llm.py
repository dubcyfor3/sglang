import sglang as sgl
import argparse
import json
import time
import random
import numpy as np
from transformers import AutoTokenizer
from dataset_utils import (
    load_sharegpt_dataset,
    load_gsm8k_dataset,
    load_humaneval_dataset,
    RANDOM_SEED,
)

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def main(args):
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
    
    # Initialize engine (standard LLM, no DLLM)
    llm = sgl.Engine(
        model_path=args.model_path,
        max_running_requests=args.max_running_requests,
        trust_remote_code=True
    )

    # Prepare sampling parameters
    # Use dataset-specific output lengths if available, otherwise use default
    if output_lens and len(output_lens) == len(prompts) and all(olen is not None for olen in output_lens):
        # Create per-request sampling parameters with dataset-specific output lengths
        sampling_params_list = []
        for output_len in output_lens:
            params = {
                "temperature": getattr(args, 'temperature', 0),
                "max_new_tokens": output_len,
            }
            if args.dataset == "gsm8k":
                params["stop"] = ["Question", "Assistant:", "<|separator|>"]
            sampling_params_list.append(params)
        sampling_params = sampling_params_list
        print(f"Using per-request output lengths (dataset-specific)")
    else:
        # Fallback to single sampling params for all requests
        sampling_params = {
            "temperature": getattr(args, 'temperature', 0),
            "max_new_tokens": getattr(args, 'max_new_tokens', 1024),
        }
        if args.dataset == "gsm8k":
            sampling_params["stop"] = ["Question", "Assistant:", "<|separator|>"]
        print(f"Using default max_new_tokens: {sampling_params['max_new_tokens']}")

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
            "throughput": total_tokens / latency if total_tokens > 0 else 0,
            "total_output_tokens": total_tokens,
        }
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate standard LLM on various datasets")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="Path to the model (default: Llama 3 8B)")
    parser.add_argument("--max_running_requests", type=int, default=1)
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="sharegpt", 
                       choices=["sharegpt", "gsm8k", "humaneval"],
                       help="Dataset to evaluate on")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to evaluate")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    
    # Output
    parser.add_argument("--output_file", type=str, default=None,
                       help="Path to save evaluation results JSON")
    
    args = parser.parse_args()
    main(args)

