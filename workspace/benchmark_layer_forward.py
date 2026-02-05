import argparse
import os
import time
import torch
import numpy as np
import yaml
from collections import defaultdict
from transformers import AutoTokenizer
from typing import Optional

# SGLang imports
import sglang as sgl


def set_algorithm_config(config_path, block_size, k=None, threshold=None):
    """Set the algorithm configuration in a YAML config file."""
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
    """Create a prompt that tokenizes to exactly target_length tokens."""
    prompt = base_text
    current_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    current_length = len(current_ids)
    
    if current_length == target_length:
        return prompt
    
    if current_length < target_length:
        words = base_text.split()
        word_idx = 0
        while current_length < target_length:
            if word_idx >= len(words):
                word_idx = 0
            prompt += words[word_idx] + " "
            word_idx += 1
            current_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
            current_length = len(current_ids)
            if current_length > target_length * 2:
                break
        
        while current_length > target_length:
            words_list = prompt.split()
            if len(words_list) > 1:
                prompt = " ".join(words_list[:-1]) + " "
            else:
                break
            current_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
            current_length = len(current_ids)
    else:
        words = prompt.split()
        while current_length > target_length and len(words) > 1:
            words = words[:-1]
            prompt = " ".join(words) + " "
            current_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
            current_length = len(current_ids)
    
    return prompt


class LayerProfiler:
    """Profiler to measure latency of each model component."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.hooks = []
        self.start_times = {}
        
    def _register_module_hook(self, module, name):
        """Register hooks on a module."""
        self.start_times[name] = 0
        
        def pre_hook(module, input):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_times[name] = time.perf_counter()
        
        def post_hook(module, input, output):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            if self.start_times[name] > 0:
                self.timings[name].append(end_time - self.start_times[name])
        
        hook_handle = module.register_forward_hook(post_hook)
        pre_hook_handle = module.register_forward_pre_hook(pre_hook)
        self.hooks.append((hook_handle, pre_hook_handle))
    
    def register_hooks(self, model):
        """Register hooks on all model components."""
        # Embeddings
        if hasattr(model, 'model') and hasattr(model.model, 'word_embeddings'):
            self._register_module_hook(model.model.word_embeddings, "embedding")
        
        # Layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for i, layer in enumerate(model.model.layers):
                layer_prefix = f"layer_{i}"
                
                # Input layer norm
                if hasattr(layer, 'input_layernorm'):
                    self._register_module_hook(layer.input_layernorm, f"{layer_prefix}_input_layernorm")
                
                # Attention
                if hasattr(layer, 'attention'):
                    self._register_module_hook(layer.attention, f"{layer_prefix}_attention")
                
                # Post attention layer norm
                if hasattr(layer, 'post_attention_layernorm'):
                    self._register_module_hook(layer.post_attention_layernorm, f"{layer_prefix}_post_attn_layernorm")
                
                # MLP
                if hasattr(layer, 'mlp'):
                    self._register_module_hook(layer.mlp, f"{layer_prefix}_mlp")
        
        # Final norm
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            self._register_module_hook(model.model.norm, "final_norm")
        
        # LM head
        if hasattr(model, 'lm_head'):
            self._register_module_hook(model.lm_head, "lm_head")
        
        # Logits processor
        if hasattr(model, 'logits_processor'):
            self._register_module_hook(model.logits_processor, "logits_processor")
    
    def start_iteration(self):
        """Start a new iteration."""
        self.start_times = {}
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook_handle, pre_hook_handle in self.hooks:
            hook_handle.remove()
            pre_hook_handle.remove()
        self.hooks = []
    
    def get_statistics(self):
        """Get statistics for all timings."""
        stats = {}
        for name, times in self.timings.items():
            if times:
                times_array = np.array(times)
                stats[name] = {
                    'avg': np.mean(times_array) * 1000,  # Convert to ms
                    'median': np.median(times_array) * 1000,
                    'min': np.min(times_array) * 1000,
                    'max': np.max(times_array) * 1000,
                    'std': np.std(times_array) * 1000,
                    'count': len(times),
                }
        return stats




def benchmark_layer_forward(
    model_path: str,
    batch_size: int = 1,
    input_length: int = 256,
    output_length: int = 256,
    num_warmup: int = 3,
    num_iterations: int = 10,
    disable_cuda_graph: bool = True,
    dllm_algorithm: Optional[str] = None,
    dllm_algorithm_config: Optional[str] = None,
    block_size: int = 32,
    k: Optional[int] = None,
    confidence_threshold: Optional[float] = None,
):
    """
    Benchmark forward pass time for each layer/component in LLaDA2 model.
    Uses SGLang Engine's generate() function.
    For single GPU use only.
    
    Note: To profile per-layer latency, CUDA graph must be disabled.
    """
    # Load tokenizer
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Create prompts
    print(f"Creating prompts of length {input_length} tokens...")
    base_prompt = create_prompt_of_length(tokenizer, input_length)
    prompts = [base_prompt] * batch_size
    
    # Initialize SGLang engine
    print(f"\nInitializing SGLang engine...")
    print(f"  Model: {model_path}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Input Length: {input_length} tokens")
    print(f"  Output Length: {output_length} tokens")
    print(f"  Disable CUDA Graph: {disable_cuda_graph}")
    if dllm_algorithm:
        print(f"  DLLM Algorithm: {dllm_algorithm}")
        if dllm_algorithm_config:
            print(f"  DLLM Algorithm Config: {dllm_algorithm_config}")
        print(f"  Block Size: {block_size}")
        if k is not None:
            print(f"  TopK k: {k}")
        if confidence_threshold is not None:
            print(f"  Confidence Threshold: {confidence_threshold}")
    
    # Enable layer timing recording via environment variable
    # This will be picked up in the scheduler subprocess
    import os
    os.environ["SGLANG_ENABLE_LAYER_TIMING"] = "1"
    print(f"  Layer timing recording: ENABLED (via environment variable)")
    
    engine_kwargs = {
        "model_path": model_path,
        "max_running_requests": batch_size,
        "trust_remote_code": True,
        "disable_cuda_graph": disable_cuda_graph,
        "log_level": "error",
        "mem_fraction_static": 0.5,
    }
    
    # Add DLLM parameters if provided
    if dllm_algorithm:
        engine_kwargs["dllm_algorithm"] = dllm_algorithm
        if dllm_algorithm_config:
            engine_kwargs["dllm_algorithm_config"] = dllm_algorithm_config
        # Force flashinfer backend for DLLM (even when CUDA graph is disabled)
        engine_kwargs["attention_backend"] = "flashinfer"
    
    llm = sgl.Engine(**engine_kwargs)
    
    # Try to access the model for profiling
    # The model is in a subprocess, so we need to access it through the scheduler
    model = None
    try:
        # Try to get model from scheduler process
        if hasattr(llm, 'scheduler_info') and hasattr(llm.scheduler_info, 'model_runner'):
            model_runner = llm.scheduler_info.model_runner
            if hasattr(model_runner, 'model'):
                model = model_runner.model
        elif hasattr(llm, '_model'):
            model = llm._model
    except Exception as e:
        print(f"Warning: Could not access model directly: {e}")
    
    if model is None:
        print("Note: Model is in a subprocess. Per-layer profiling may be limited.")
    
    # Create profiler and register hooks if model is accessible
    profiler = LayerProfiler()
    if model is not None:
        print("Model accessible, registering profiling hooks...")
        profiler.register_hooks(model)
    else:
        print("Warning: Model not directly accessible. Per-layer profiling disabled.")
        print("Only total generation time will be measured.")
    
    # Prepare sampling parameters
    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": output_length,
        "ignore_eos": True,  # Generate exactly output_length tokens
    }
    
    print(f"\nWarming up ({num_warmup} iterations)...")
    for i in range(num_warmup):
        _ = llm.generate(prompts, sampling_params)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    print(f"\nBenchmarking layer forward pass ({num_iterations} iterations)...")
    
    for i in range(num_iterations):
        if model is not None:
            profiler.start_iteration()
        
        # Synchronize before each iteration
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Run generation (hooks will capture timings if model is accessible)
        _ = llm.generate(prompts, sampling_params)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/{num_iterations} completed")
    
    # Get statistics
    stats = {}
    if model is not None:
        stats = profiler.get_statistics()
        # Remove hooks
        profiler.remove_hooks()
    
    # Get layer timing statistics from the scheduler process
    layer_timing_stats = {}
    try:
        server_info = llm.get_server_info()
        internal_states = server_info.get("internal_states", [])
        if internal_states and len(internal_states) > 0:
            state = internal_states[0]
            if "layer_timing_stats" in state:
                layer_timing_stats = state["layer_timing_stats"]
    except Exception as e:
        print(f"Warning: Could not retrieve layer timing stats: {e}")
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Layer-by-Layer Forward Pass Benchmark Results")
    print(f"{'='*80}")
    
    # Print layer timing stats if available (aggregated by component type)
    if layer_timing_stats:
        print(f"\n{'='*80}")
        print(f"Per-Component Timing Statistics (aggregated across all layers)")
        print(f"{'='*80}")
        print(f"{'Component':<40} {'Avg (ms)':<12} {'Count':<10} {'Total (ms)':<12}")
        print(f"{'-'*80}")
        
        # Sort components: qkv, attention, dense, mlp, router, experts
        component_order = {
            'qkv': 0, 
            'attention': 1, 
            'dense': 2, 
            'mlp': 3, 
            'router': 4, 
            'experts': 5
        }
        sorted_timing = sorted(layer_timing_stats.items(), 
                              key=lambda x: (component_order.get(x[0].lower(), 99), x[0]))
        
        total_time = 0
        for name, stat in sorted_timing:
            avg_ms = stat.get('avg', 0)
            count = stat.get('count', 0)
            total_ms = stat.get('sum', 0)
            print(f"{name:<40} {avg_ms:<12.3f} {count:<10} {total_ms:<12.3f}")
            total_time += total_ms
        
        print(f"{'-'*80}")
        print(f"{'Total (sum of all components)':<40} {'':<12} {'':<10} {total_time:<12.3f}")
        print(f"{'='*80}\n")
        
        # Merge with hook-based stats if available
        for name, stat in layer_timing_stats.items():
            if name not in stats:
                stats[name] = {
                    'avg': stat.get('avg', 0),
                    'median': stat.get('avg', 0),  # Use avg as median approximation
                    'min': stat.get('avg', 0),
                    'max': stat.get('avg', 0),
                    'std': 0.0,
                    'count': stat.get('count', 0),
                }
    
    if stats:
        print(f"{'Component':<40} {'Avg (ms)':<12} {'Median (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'Std (ms)':<12}")
        print(f"{'-'*80}")
        
        # Sort by layer number for better readability
        def get_layer_num(name):
            if 'layer_' in name:
                try:
                    return int(name.split('_')[1])
                except:
                    return -1
            # Order: embedding, layers, final_norm, lm_head, logits_processor
            order = {'embedding': 0, 'final_norm': 1000, 'lm_head': 1001, 'logits_processor': 1002}
            return order.get(name.split('_')[0], 500)
        
        sorted_stats = sorted(stats.items(), key=lambda x: (get_layer_num(x[0]), x[0]))
        
        total_time = 0
        for name, stat in sorted_stats:
            print(f"{name:<40} {stat['avg']:<12.3f} {stat['median']:<12.3f} {stat['min']:<12.3f} {stat['max']:<12.3f} {stat['std']:<12.3f}")
            total_time += stat['avg']
        
        print(f"{'-'*80}")
        print(f"{'Total (sum of components)':<40} {total_time:<12.3f}")
        print(f"{'='*80}\n")
        
        # Save to CSV
        csv_file = "layer_benchmark_results.csv"
        with open(csv_file, 'w') as f:
            f.write("component,avg_ms,median_ms,min_ms,max_ms,std_ms,count\n")
            for name, stat in sorted_stats:
                f.write(f"{name},{stat['avg']:.3f},{stat['median']:.3f},{stat['min']:.3f},{stat['max']:.3f},{stat['std']:.3f},{stat['count']}\n")
        
        print(f"Results saved to {csv_file}")
    else:
        print("Per-layer profiling not available (model not accessible).")
        print("Only total generation time was measured.")
        print(f"{'='*80}\n")
    
    # Cleanup
    llm.shutdown()
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLaDA2 layer-by-layer forward pass time using SGLang Engine")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="inclusionAI/LLaDA2.0-mini",
                       help="Path to the model")
    
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
    parser.add_argument("--disable_cuda_graph", action="store_true",
                       help="Disable CUDA graph (required for per-layer profiling)")

    
    
    # DLLM arguments
    parser.add_argument("--dllm_algorithm", type=str, default="TopK",
                       help="The diffusion LLM algorithm (e.g., 'LowConfidence')")
    parser.add_argument("--dllm_algorithm_config", type=str, default="workspace/config.yaml",
                       help="Path to DLLM algorithm configuration YAML file")
    parser.add_argument("--block_size", type=int, default=32,
                       help="DLLM block size")
    parser.add_argument("--k", type=int, default=1,
                       help="TopK algorithm: number of tokens to unmask per iteration")
    parser.add_argument("--confidence_threshold", type=float, default=0.90,
                       help="LowConfidence algorithm: confidence threshold")
    
    args = parser.parse_args()

    args.disable_cuda_graph = True
    
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
    
    benchmark_layer_forward(
        model_path=args.model_path,
        batch_size=args.batch_size,
        input_length=args.input_length,
        output_length=args.output_length,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        disable_cuda_graph=args.disable_cuda_graph,
        dllm_algorithm=args.dllm_algorithm,
        dllm_algorithm_config=args.dllm_algorithm_config,
        block_size=args.block_size,
        k=args.k,
        confidence_threshold=args.confidence_threshold,
    )
