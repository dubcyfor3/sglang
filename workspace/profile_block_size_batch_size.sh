#!/bin/bash

# Script to profile DLLM performance with different block sizes and batch sizes
# Extracts forward time metrics and saves to CSV

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/benchmark_forward.py"
OUTPUT_CSV="${SCRIPT_DIR}/block_size_batch_size_profile_new.csv"
LOG_DIR="${SCRIPT_DIR}/profile_logs"

# Default values (can be overridden via environment variables)
BLOCK_SIZES="${BLOCK_SIZES:-1 2 4}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
MODEL_PATH="${MODEL_PATH:-inclusionAI/LLaDA2.0-mini}"
DLLM_ALGORITHM="${DLLM_ALGORITHM:-TopK}"
DLLM_CONFIG="${DLLM_CONFIG:-workspace/config.yaml}"
K="${K:-1}"
NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
NUM_WARMUP="${NUM_WARMUP:-3}"

# Validate Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "Error: Python script not found: ${PYTHON_SCRIPT}"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found in PATH"
    exit 1
fi

# Create log directory
mkdir -p "${LOG_DIR}"

# CSV header
echo "block_size,batch_size,avg_time_ms,median_time_ms,min_time_ms,max_time_ms,std_time_ms,throughput_tokens_per_sec,forward_time_avg_ms" > "${OUTPUT_CSV}"

# Function to extract value from output
extract_value() {
    local pattern="$1"
    local output="$2"
    # Use grep with Perl regex to extract capture group, then extract just the number
    local value=$(echo "$output" | grep -oP "${pattern}" | grep -oE "[0-9]+\.[0-9]+|[0-9]+" | head -1)
    if [ -z "$value" ]; then
        echo "N/A"
    else
        echo "$value"
    fi
}

# Function to run benchmark and extract metrics
run_profile() {
    local block_size="$1"
    local batch_size="$2"
    local log_file="${LOG_DIR}/block_${block_size}_batch_${batch_size}.log"
    
    echo "=========================================="
    echo "Profiling: block_size=${block_size}, batch_size=${batch_size}"
    echo "=========================================="
    
    # Run benchmark and capture output
    python "${PYTHON_SCRIPT}" \
        --model_path "${MODEL_PATH}" \
        --dllm_algorithm "${DLLM_ALGORITHM}" \
        --dllm_algorithm_config "${DLLM_CONFIG}" \
        --block_size "${block_size}" \
        --k "${K}" \
        --batch_size "${batch_size}" \
        --num_iterations "${NUM_ITERATIONS}" \
        --num_warmup "${NUM_WARMUP}" \
        2>&1 | tee "${log_file}"
    
    local output=$(cat "${log_file}")
    
    # Extract metrics from output using sed for more reliable extraction
    local avg_time=$(echo "$output" | grep "Average time:" | sed -E 's/.*Average time:[[:space:]]+([0-9]+\.[0-9]+)[[:space:]]+ms.*/\1/' | head -1)
    local median_time=$(echo "$output" | grep "Median time:" | sed -E 's/.*Median time:[[:space:]]+([0-9]+\.[0-9]+)[[:space:]]+ms.*/\1/' | head -1)
    local min_time=$(echo "$output" | grep "Min time:" | sed -E 's/.*Min time:[[:space:]]+([0-9]+\.[0-9]+)[[:space:]]+ms.*/\1/' | head -1)
    local max_time=$(echo "$output" | grep "Max time:" | sed -E 's/.*Max time:[[:space:]]+([0-9]+\.[0-9]+)[[:space:]]+ms.*/\1/' | head -1)
    local std_time=$(echo "$output" | grep "Std deviation:" | sed -E 's/.*Std deviation:[[:space:]]+([0-9]+\.[0-9]+)[[:space:]]+ms.*/\1/' | head -1)
    local throughput=$(echo "$output" | grep "Throughput:" | sed -E 's/.*Throughput:[[:space:]]+([0-9]+\.[0-9]+)[[:space:]]+tokens\/second.*/\1/' | head -1)
    local forward_time_avg=$(echo "$output" | grep "Forward time (avg):" | sed -E 's/.*Forward time \(avg\):[[:space:]]+([0-9]+\.[0-9]+)[[:space:]]+ms.*/\1/' | head -1)
    
    # Handle empty values
    [ -z "$avg_time" ] && avg_time="N/A"
    [ -z "$median_time" ] && median_time="N/A"
    [ -z "$min_time" ] && min_time="N/A"
    [ -z "$max_time" ] && max_time="N/A"
    [ -z "$std_time" ] && std_time="N/A"
    [ -z "$throughput" ] && throughput="N/A"
    [ -z "$forward_time_avg" ] && forward_time_avg="N/A"
    
    # Write to CSV
    echo "${block_size},${batch_size},${avg_time},${median_time},${min_time},${max_time},${std_time},${throughput},${forward_time_avg}" >> "${OUTPUT_CSV}"
    
    echo ""
    echo "Results saved: block_size=${block_size}, batch_size=${batch_size}"
    echo "  Avg time: ${avg_time} ms"
    echo "  Throughput: ${throughput} tokens/sec"
    echo ""
}

# Main profiling loop
echo "Starting profiling..."
echo "Block sizes: ${BLOCK_SIZES}"
echo "Batch sizes: ${BATCH_SIZES}"
echo "Output CSV: ${OUTPUT_CSV}"
echo ""

# Loop over block sizes and batch sizes
for block_size in ${BLOCK_SIZES}; do
    for batch_size in ${BATCH_SIZES}; do
        run_profile "${block_size}" "${batch_size}"
        
        # Small delay between runs
        sleep 2
    done
done

echo "=========================================="
echo "Profiling complete!"
echo "Results saved to: ${OUTPUT_CSV}"
echo "Logs saved to: ${LOG_DIR}"
echo "=========================================="
