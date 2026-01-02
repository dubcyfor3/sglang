"""
Dataset loading and evaluation utilities for DLLM evaluation.
"""

import os
import random
import re
import ast
from typing import List, Tuple, Optional
from transformers import PreTrainedTokenizerBase
from sglang.utils import download_and_cache_file, read_jsonl
from sglang.bench_serving import (
    sample_sharegpt_requests,
    SHAREGPT_REPO_ID,
    SHAREGPT_FILENAME,
    download_and_cache_hf_file,
)

# Constants
INVALID = -9999999
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

# Fixed seed for reproducibility - ensures deterministic dataset sampling across runs
RANDOM_SEED = 42


def load_sharegpt_dataset(args, tokenizer: PreTrainedTokenizerBase, num_samples: int):
    """Load ShareGPT dataset."""
    # Set random seed for reproducibility (before calling sample_sharegpt_requests which shuffles)
    random.seed(RANDOM_SEED)
    
    dataset_path = args.sharegpt_path if hasattr(args, 'sharegpt_path') else ""
    
    # Download if necessary
    if not dataset_path or not os.path.isfile(dataset_path):
        dataset_path = download_and_cache_hf_file(
            repo_id=SHAREGPT_REPO_ID,
            filename=SHAREGPT_FILENAME,
        )
    
    requests = sample_sharegpt_requests(
        dataset_path=dataset_path,
        num_requests=num_samples,
        tokenizer=tokenizer,
        fixed_output_len=getattr(args, 'sharegpt_output_len', None),
        context_len=getattr(args, 'sharegpt_context_len', None),
        apply_chat_template=getattr(args, 'apply_chat_template', False),
    )
    
    # With fixed seed set above, the shuffle in sample_sharegpt_requests should be deterministic
    # requests already contains exactly num_samples samples in deterministic order
    
    prompts = [req.prompt for req in requests]
    return prompts, requests


def get_answer_value(answer_str: str) -> int:
    """Extract numeric answer from GSM8K answer string."""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def load_gsm8k_dataset(args, num_samples: int, tokenizer: Optional[PreTrainedTokenizerBase] = None):
    """Load GSM8K dataset."""
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    data_path = getattr(args, 'gsm8k_path', None)
    if not data_path or not os.path.isfile(data_path):
        data_path = download_and_cache_file(GSM8K_URL)
    
    lines = list(read_jsonl(data_path))
    num_shots = getattr(args, 'num_shots', 5)
    
    # Ensure deterministic order by using first N samples (already deterministic, but explicit)
    
    # Get few-shot examples
    def get_one_example(lines, i, include_answer):
        ret = "Question: " + lines[i]["question"] + "\nAnswer:"
        if include_answer:
            ret += " " + lines[i]["answer"]
        return ret
    
    def get_few_shot_examples(lines, k):
        ret = ""
        for i in range(k):
            ret += get_one_example(lines, i, True) + "\n\n"
        return ret
    
    few_shot_examples = get_few_shot_examples(lines, num_shots)
    
    # Prepare questions
    prompts = []
    labels = []
    output_lens = []
    for i in range(min(num_samples, len(lines))):
        question = get_one_example(lines, i, False)
        prompts.append(few_shot_examples + question)
        labels.append(get_answer_value(lines[i]["answer"]))
        
        # Calculate output length from the answer text
        if tokenizer:
            answer_text = lines[i]["answer"]
            output_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
            output_lens.append(len(output_tokens))
        else:
            output_lens.append(None)
    
    return prompts, (labels, output_lens)


def load_humaneval_dataset(args, num_samples: int, tokenizer: Optional[PreTrainedTokenizerBase] = None):
    """Load HumanEval dataset."""
    try:
        from human_eval.data import read_problems
    except ImportError:
        raise ImportError("Please install human-eval: pip install human-eval")
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    problems = read_problems()
    problems_list = list(problems.values())
    
    # Sort by problem key to ensure deterministic order before sampling
    problems_list = sorted(problems_list, key=lambda x: x.get("task_id", ""))
    
    if num_samples:
        # Use seeded random to ensure deterministic sampling
        rng = random.Random(RANDOM_SEED)
        problems_list = rng.sample(problems_list, min(num_samples, len(problems_list)))
        # Sort again after sampling to ensure consistent order
        problems_list = sorted(problems_list, key=lambda x: x.get("task_id", ""))
    
    instruction = "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n"
    
    prompts = []
    labels = []
    output_lens = []
    for problem in problems_list:
        prompt = instruction + problem["prompt"]
        prompts.append(prompt)
        labels.append(problem)  # Store full problem dict for evaluation
        
        # Calculate output length from the canonical solution (test)
        # Use the test as a proxy for expected output length
        if tokenizer:
            # Extract the function body from the test
            test_code = problem.get("test", "")
            # Estimate output length based on test complexity
            # For now, use a reasonable estimate based on prompt length
            # Or use the canonical solution if available
            canonical_solution = problem.get("canonical_solution", "")
            if canonical_solution:
                solution_tokens = tokenizer.encode(canonical_solution, add_special_tokens=False)
                output_lens.append(len(solution_tokens))
            else:
                # Fallback: estimate based on prompt length
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                output_lens.append(max(50, len(prompt_tokens) // 2))  # Rough estimate
        else:
            output_lens.append(None)
    
    return prompts, (labels, output_lens)


def evaluate_gsm8k(outputs: List, labels: List[int]) -> dict:
    """Evaluate GSM8K predictions."""
    preds = []
    for output in outputs:
        if isinstance(output, dict):
            text = output.get("text", "")
        else:
            text = str(output)
        preds.append(get_answer_value(text))
    
    if len(preds) != len(labels):
        print(f"Warning: Mismatch between predictions ({len(preds)}) and labels ({len(labels)})")
        min_len = min(len(preds), len(labels))
        preds = preds[:min_len]
        labels = labels[:min_len]
    
    correct = sum(p == l for p, l in zip(preds, labels) if p != INVALID)
    accuracy = correct / len(labels) if labels else 0.0
    invalid = sum(p == INVALID for p in preds) / len(preds) if preds else 0.0
    
    return {
        "accuracy": accuracy,
        "invalid_rate": invalid,
        "num_correct": correct,
        "num_total": len(labels),
    }


def evaluate_humaneval(outputs: List, labels: List) -> dict:
    """Evaluate HumanEval predictions."""
    try:
        from human_eval.execution import check_correctness
        from human_eval.evaluation import estimate_pass_at_k
    except ImportError:
        raise ImportError("Please install human-eval: pip install human-eval")
    
    def find_code(completion: str) -> str:
        completion = completion or ""
        pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
        matches = pattern.findall(completion)
        extracted = matches[0] if matches else completion
        # Remove function signature if present
        if ":\n    " in extracted:
            extracted = extracted[extracted.find(":\n    ") + 2:]
        return extracted
    
    if len(outputs) != len(labels):
        print(f"Warning: Mismatch between outputs ({len(outputs)}) and labels ({len(labels)})")
        min_len = min(len(outputs), len(labels))
        outputs = outputs[:min_len]
        labels = labels[:min_len]
    
    passed = []
    for output, problem in zip(outputs, labels):
        if isinstance(output, dict):
            text = output.get("text", "")
        else:
            text = str(output)
        code = find_code(text)
        result = check_correctness(problem, code, timeout=3.0)
        passed.append(int(result["passed"]))
    
    total = len(passed)
    correct = sum(passed)
    pass_at_1 = estimate_pass_at_k([total], [correct], 1)[0] if total >= 1 else 0.0
    
    return {
        "pass@1": pass_at_1,
        "num_correct": correct,
        "num_total": total,
    }


if __name__ == '__main__':
    from transformers import AutoTokenizer
    
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    import numpy as np
    np.random.seed(RANDOM_SEED)
    
    # Use a default model for tokenization
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Create a simple args object
    class DatasetArgs:
        pass
    dataset_args = DatasetArgs()
    
    num_samples = 100
    
    print("\n" + "="*60)
    print("Dataset Statistics (100 samples each)")
    print("="*60)
    
    # ShareGPT
    print("\n[ShareGPT]")
    try:
        prompts, dataset_info = load_sharegpt_dataset(dataset_args, tokenizer, num_samples)
        input_lens = []
        output_lens = []
        for req in dataset_info:
            prompt_tokens = tokenizer.encode(req.prompt, add_special_tokens=False)
            input_lens.append(len(prompt_tokens))
            output_lens.append(req.output_len)
        
        avg_input = sum(input_lens) / len(input_lens) if input_lens else 0
        avg_output = sum(output_lens) / len(output_lens) if output_lens else 0
        print(f"  Samples: {len(prompts)}")
        print(f"  Avg input length: {avg_input:.1f} tokens")
        print(f"  Avg output length: {avg_output:.1f} tokens")
        print(f"  Min input: {min(input_lens)}, Max input: {max(input_lens)}")
        print(f"  Min output: {min(output_lens)}, Max output: {max(output_lens)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # GSM8K
    print("\n[GSM8K]")
    try:
        result = load_gsm8k_dataset(dataset_args, num_samples, tokenizer)
        prompts, (labels, output_lens) = result
        input_lens = []
        for prompt in prompts:
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            input_lens.append(len(prompt_tokens))
        
        avg_input = sum(input_lens) / len(input_lens) if input_lens else 0
        avg_output = sum(output_lens) / len(output_lens) if output_lens else 0
        print(f"  Samples: {len(prompts)}")
        print(f"  Avg input length: {avg_input:.1f} tokens")
        print(f"  Avg output length: {avg_output:.1f} tokens")
        print(f"  Min input: {min(input_lens)}, Max input: {max(input_lens)}")
        print(f"  Min output: {min(output_lens)}, Max output: {max(output_lens)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # HumanEval
    print("\n[HumanEval]")
    try:
        result = load_humaneval_dataset(dataset_args, num_samples, tokenizer)
        prompts, (labels, output_lens) = result
        input_lens = []
        for prompt in prompts:
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            input_lens.append(len(prompt_tokens))
        
        avg_input = sum(input_lens) / len(input_lens) if input_lens else 0
        avg_output = sum(output_lens) / len(output_lens) if output_lens else 0
        print(f"  Samples: {len(prompts)}")
        print(f"  Avg input length: {avg_input:.1f} tokens")
        print(f"  Avg output length: {avg_output:.1f} tokens")
        print(f"  Min input: {min(input_lens)}, Max input: {max(input_lens)}")
        print(f"  Min output: {min(output_lens)}, Max output: {max(output_lens)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "="*60)