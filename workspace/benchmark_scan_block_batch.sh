#!/bin/bash

# Script to scan through different block sizes and batch sizes
# and capture component timing statistics to CSV
#
# Usage:
#   ./benchmark_scan_block_batch.sh
#
# Environment variables (optional):
#   MODEL_PATH - Model path (default: inclusionAI/LLaDA2.0-mini)
#   DLLM_ALGORITHM - Algorithm to use (default: TopK)
#   BLOCK_SIZES - Space-separated list of block sizes (default: "16 32 64")
#   BATCH_SIZES - Space-separated list of batch sizes (default: "1 2 4")
#   K_VALUES - Space-separated list of k values for TopK (default: "1 2")
#   OUTPUT_CSV - Output CSV file path (default: benchmark_scan_results.csv)
#
# Example:
#   BLOCK_SIZES="16 32 64" BATCH_SIZES="1 2 4" ./benchmark_scan_block_batch.sh

# Don't use set -e globally, handle errors explicitly in loops
set -o pipefail  # Exit on pipe failures

# Configuration
MODEL_PATH="${MODEL_PATH:-inclusionAI/LLaDA2.0-mini}"
DLLM_ALGORITHM="${DLLM_ALGORITHM:-TopK}"
DLLM_CONFIG="${DLLM_CONFIG:-workspace/config.yaml}"
INPUT_LENGTH="${INPUT_LENGTH:-256}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-256}"
NUM_WARMUP="${NUM_WARMUP:-3}"
NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_layer_forward.py"

# Create output directory
OUTPUT_DIR="${SCRIPT_DIR}/output"
mkdir -p "$OUTPUT_DIR"

# Set up output files with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_LOG="${OUTPUT_DIR}/benchmark_scan_${TIMESTAMP}.log"
OUTPUT_ERR="${OUTPUT_DIR}/benchmark_scan_${TIMESTAMP}.err"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_DIR}/benchmark_scan_results_${TIMESTAMP}.csv}"

# Redirect all output to log files
exec > >(tee -a "$OUTPUT_LOG")
exec 2> >(tee -a "$OUTPUT_ERR" >&2)

# Block sizes to test (space-separated)
BLOCK_SIZES="${BLOCK_SIZES:-1 2 4 8 16 32}"

# Batch sizes to test (space-separated)
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"

# K values for TopK algorithm (if using TopK)
K_VALUES="${K_VALUES:-1}"

echo "=========================================="
echo "Benchmark Scan: Block Size & Batch Size"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Algorithm: $DLLM_ALGORITHM"
echo "Block Sizes: $BLOCK_SIZES"
echo "Batch Sizes: $BATCH_SIZES"
echo "Output CSV: $OUTPUT_CSV"
echo "Output Log: $OUTPUT_LOG"
echo "Error Log: $OUTPUT_ERR"
echo "=========================================="
echo ""

# Initialize CSV file with header
CSV_HEADER="block_size,batch_size,k,input_length,output_length,qkv_avg_ms,qkv_count,qkv_total_ms,attention_avg_ms,attention_count,attention_total_ms,dense_avg_ms,dense_count,dense_total_ms,mlp_avg_ms,mlp_count,mlp_total_ms,router_avg_ms,router_count,router_total_ms,experts_avg_ms,experts_count,experts_total_ms,total_ms"
echo "$CSV_HEADER" > "$OUTPUT_CSV"

# Function to extract timing stats from benchmark output
# Uses Python helper script for more reliable parsing
extract_timing_stats() {
    local output_file="$1"
    local parse_script="${SCRIPT_DIR}/parse_timing_stats.py"
    
    # Try using Python parser first (more reliable)
    if [[ -f "$parse_script" ]] && command -v python3 &> /dev/null; then
        local json_output=$(python3 "$parse_script" "$output_file" 2>/dev/null)
        if [[ -n "$json_output" ]] && [[ "$json_output" != "{}" ]]; then
            # Parse JSON and convert to our format
            local stats=()
            for component in qkv attention dense mlp router experts; do
                local avg=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('$component', {}).get('avg', '0'))" 2>/dev/null)
                local count=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('$component', {}).get('count', '0'))" 2>/dev/null)
                local total=$(echo "$json_output" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('$component', {}).get('total', '0'))" 2>/dev/null)
                
                if [[ -n "$avg" ]] && [[ "$avg" != "None" ]]; then
                    stats+=("${component}:${avg}:${count}:${total}")
                fi
            done
            echo "${stats[@]}"
            return 0
        fi
    fi
    
    # Fallback to shell-based parsing
    local stats=()
    local in_stats_section=false
    local found_header=false
    
    while IFS= read -r line; do
        if [[ "$line" == *"Per-Component Timing Statistics"* ]]; then
            in_stats_section=true
            continue
        fi
        
        if [[ "$in_stats_section" == true ]] && [[ "$line" == *"Component"* ]] && [[ "$line" == *"Avg (ms)"* ]]; then
            found_header=true
            continue
        fi
        
        if [[ "$found_header" == true ]] && [[ "$line" == *"---"* ]]; then
            continue
        fi
        
        if [[ "$in_stats_section" == true ]] && [[ "$found_header" == true ]] && [[ "$line" == *"===="* ]]; then
            break
        fi
        
        if [[ "$in_stats_section" == true ]] && [[ "$found_header" == true ]] && [[ -n "$line" ]]; then
            local component=$(echo "$line" | awk '{print $1}')
            
            # Extract all component types: qkv, attention, dense, mlp, router, experts
            if [[ "$component" == "qkv" ]] || [[ "$component" == "attention" ]] || [[ "$component" == "dense" ]] || [[ "$component" == "mlp" ]] || [[ "$component" == "router" ]] || [[ "$component" == "experts" ]]; then
                local avg_ms=$(echo "$line" | awk '{print $2}')
                local count=$(echo "$line" | awk '{print $3}')
                local total_ms=$(echo "$line" | awk '{print $4}')
                
                if [[ "$avg_ms" =~ ^[0-9]+\.?[0-9]*$ ]] && [[ "$count" =~ ^[0-9]+$ ]]; then
                    stats+=("${component}:${avg_ms}:${count}:${total_ms}")
                fi
            fi
        fi
    done < "$output_file"
    
    echo "${stats[@]}"
}

# Function to get value from stats array
get_stat_value() {
    local component="$1"
    local field="$2"  # avg, count, or total
    shift 2
    local stats_array=("$@")
    
    for stat in "${stats_array[@]}"; do
        if [[ "$stat" == "${component}:"* ]]; then
            IFS=':' read -r comp avg count total <<< "$stat"
            case "$field" in
                avg) echo "$avg" ;;
                count) echo "$count" ;;
                total) echo "$total" ;;
            esac
            return 0
        fi
    done
    echo "0"  # Default if not found
}

# Counter for progress
total_runs=0
current_run=0

# Calculate total runs
for block_size in $BLOCK_SIZES; do
    for batch_size in $BATCH_SIZES; do
        if [[ "$DLLM_ALGORITHM" == "TopK" ]]; then
            for k in $K_VALUES; do
                ((total_runs++))
            done
        else
            ((total_runs++))
        fi
    done
done

echo "Total runs: $total_runs"
echo ""

# Main loop: iterate through all combinations
for block_size in $BLOCK_SIZES; do
    for batch_size in $BATCH_SIZES; do
        if [[ "$DLLM_ALGORITHM" == "TopK" ]]; then
            # For TopK, also iterate through k values
            for k in $K_VALUES; do
                ((current_run++))
                echo "[$current_run/$total_runs] Running: block_size=$block_size, batch_size=$batch_size, k=$k"
                
                # Create temporary output file
                temp_output=$(mktemp)
                
                # Run benchmark and capture output
                if python "$BENCHMARK_SCRIPT" \
                    --model_path "$MODEL_PATH" \
                    --dllm_algorithm "$DLLM_ALGORITHM" \
                    --dllm_algorithm_config "$DLLM_CONFIG" \
                    --block_size "$block_size" \
                    --batch_size "$batch_size" \
                    --k "$k" \
                    --input_length "$INPUT_LENGTH" \
                    --output_length "$OUTPUT_LENGTH" \
                    --num_warmup "$NUM_WARMUP" \
                    --num_iterations "$NUM_ITERATIONS" \
                    --disable_cuda_graph \
                    > "$temp_output" 2>&1; then
                    
                    # Extract timing statistics
                    stats_output=$(extract_timing_stats "$temp_output" 2>/dev/null || echo "")
                    
                    # Check if we got any stats
                    if [[ -z "$stats_output" ]]; then
                        echo "  ⚠ Warning: Could not extract timing stats from output"
                        echo "$block_size,$batch_size,$k,$INPUT_LENGTH,$OUTPUT_LENGTH,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" >> "$OUTPUT_CSV"
                        rm -f "$temp_output"
                        continue
                    fi
                    
                    # Convert to array (handle empty case)
                    if [[ -n "$stats_output" ]]; then
                        read -ra stats_array <<< "$stats_output"
                    else
                        stats_array=()
                    fi
                    
                    # Get values for each component (with defaults)
                    qkv_avg=$(get_stat_value "qkv" "avg" "${stats_array[@]}" || echo "0")
                    qkv_count=$(get_stat_value "qkv" "count" "${stats_array[@]}" || echo "0")
                    qkv_total=$(get_stat_value "qkv" "total" "${stats_array[@]}" || echo "0")
                    
                    attn_avg=$(get_stat_value "attention" "avg" "${stats_array[@]}" || echo "0")
                    attn_count=$(get_stat_value "attention" "count" "${stats_array[@]}" || echo "0")
                    attn_total=$(get_stat_value "attention" "total" "${stats_array[@]}" || echo "0")
                    
                    dense_avg=$(get_stat_value "dense" "avg" "${stats_array[@]}" || echo "0")
                    dense_count=$(get_stat_value "dense" "count" "${stats_array[@]}" || echo "0")
                    dense_total=$(get_stat_value "dense" "total" "${stats_array[@]}" || echo "0")
                    
                    mlp_avg=$(get_stat_value "mlp" "avg" "${stats_array[@]}" || echo "0")
                    mlp_count=$(get_stat_value "mlp" "count" "${stats_array[@]}" || echo "0")
                    mlp_total=$(get_stat_value "mlp" "total" "${stats_array[@]}" || echo "0")
                    
                    router_avg=$(get_stat_value "router" "avg" "${stats_array[@]}" || echo "0")
                    router_count=$(get_stat_value "router" "count" "${stats_array[@]}" || echo "0")
                    router_total=$(get_stat_value "router" "total" "${stats_array[@]}" || echo "0")
                    
                    experts_avg=$(get_stat_value "experts" "avg" "${stats_array[@]}" || echo "0")
                    experts_count=$(get_stat_value "experts" "count" "${stats_array[@]}" || echo "0")
                    experts_total=$(get_stat_value "experts" "total" "${stats_array[@]}" || echo "0")
                    
                    # Calculate total (use awk for floating point if bc not available)
                    if command -v bc &> /dev/null; then
                        total_ms=$(echo "$qkv_total + $attn_total + $dense_total + $mlp_total + $router_total + $experts_total" | bc 2>/dev/null || echo "0")
                    else
                        total_ms=$(awk "BEGIN {printf \"%.2f\", $qkv_total + $attn_total + $dense_total + $mlp_total + $router_total + $experts_total}" 2>/dev/null || echo "0")
                    fi
                    
                    # Write to CSV
                    echo "$block_size,$batch_size,$k,$INPUT_LENGTH,$OUTPUT_LENGTH,$qkv_avg,$qkv_count,$qkv_total,$attn_avg,$attn_count,$attn_total,$dense_avg,$dense_count,$dense_total,$mlp_avg,$mlp_count,$mlp_total,$router_avg,$router_count,$router_total,$experts_avg,$experts_count,$experts_total,$total_ms" >> "$OUTPUT_CSV"
                    
                    echo "  ✓ Success: qkv=${qkv_avg}ms, attention=${attn_avg}ms, dense=${dense_avg}ms, mlp=${mlp_avg}ms, router=${router_avg}ms, experts=${experts_avg}ms"
                else
                    echo "  ✗ Failed: Check $temp_output for details"
                    # Show last few lines of error
                    echo "  Last 10 lines of output:"
                    tail -n 10 "$temp_output" | sed 's/^/    /'
                    echo "$block_size,$batch_size,$k,$INPUT_LENGTH,$OUTPUT_LENGTH,ERROR,0,0,ERROR,0,0,ERROR,0,0,ERROR,0,0,ERROR,0,0,ERROR,0,0,ERROR" >> "$OUTPUT_CSV"
                fi
                
                # Cleanup
                rm -f "$temp_output"
                echo ""
            done
        else
            # For LowConfidence or other algorithms
            ((current_run++))
            echo "[$current_run/$total_runs] Running: block_size=$block_size, batch_size=$batch_size"
            
            # Create temporary output file
            temp_output=$(mktemp)
            
            # Run benchmark and capture output
            if python "$BENCHMARK_SCRIPT" \
                --model_path "$MODEL_PATH" \
                --dllm_algorithm "$DLLM_ALGORITHM" \
                --dllm_algorithm_config "$DLLM_CONFIG" \
                --block_size "$block_size" \
                --batch_size "$batch_size" \
                --input_length "$INPUT_LENGTH" \
                --output_length "$OUTPUT_LENGTH" \
                --num_warmup "$NUM_WARMUP" \
                --num_iterations "$NUM_ITERATIONS" \
                --disable_cuda_graph \
                > "$temp_output" 2>&1; then
                
                # Extract timing statistics
                stats_output=$(extract_timing_stats "$temp_output" 2>/dev/null || echo "")
                
                # Check if we got any stats
                if [[ -z "$stats_output" ]]; then
                    echo "  ⚠ Warning: Could not extract timing stats from output"
                    echo "$block_size,$batch_size,0,$INPUT_LENGTH,$OUTPUT_LENGTH,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" >> "$OUTPUT_CSV"
                    rm -f "$temp_output"
                    continue
                fi
                
                # Convert to array (handle empty case)
                if [[ -n "$stats_output" ]]; then
                    read -ra stats_array <<< "$stats_output"
                else
                    stats_array=()
                fi
                
                # Get values for each component (with defaults)
                qkv_avg=$(get_stat_value "qkv" "avg" "${stats_array[@]}" || echo "0")
                qkv_count=$(get_stat_value "qkv" "count" "${stats_array[@]}" || echo "0")
                qkv_total=$(get_stat_value "qkv" "total" "${stats_array[@]}" || echo "0")
                
                attn_avg=$(get_stat_value "attention" "avg" "${stats_array[@]}" || echo "0")
                attn_count=$(get_stat_value "attention" "count" "${stats_array[@]}" || echo "0")
                attn_total=$(get_stat_value "attention" "total" "${stats_array[@]}" || echo "0")
                
                dense_avg=$(get_stat_value "dense" "avg" "${stats_array[@]}" || echo "0")
                dense_count=$(get_stat_value "dense" "count" "${stats_array[@]}" || echo "0")
                dense_total=$(get_stat_value "dense" "total" "${stats_array[@]}" || echo "0")
                
                mlp_avg=$(get_stat_value "mlp" "avg" "${stats_array[@]}" || echo "0")
                mlp_count=$(get_stat_value "mlp" "count" "${stats_array[@]}" || echo "0")
                mlp_total=$(get_stat_value "mlp" "total" "${stats_array[@]}" || echo "0")
                
                router_avg=$(get_stat_value "router" "avg" "${stats_array[@]}" || echo "0")
                router_count=$(get_stat_value "router" "count" "${stats_array[@]}" || echo "0")
                router_total=$(get_stat_value "router" "total" "${stats_array[@]}" || echo "0")
                
                experts_avg=$(get_stat_value "experts" "avg" "${stats_array[@]}" || echo "0")
                experts_count=$(get_stat_value "experts" "count" "${stats_array[@]}" || echo "0")
                experts_total=$(get_stat_value "experts" "total" "${stats_array[@]}" || echo "0")
                
                # Calculate total (use awk for floating point if bc not available)
                if command -v bc &> /dev/null; then
                    total_ms=$(echo "$qkv_total + $attn_total + $dense_total + $mlp_total + $router_total + $experts_total" | bc 2>/dev/null || echo "0")
                else
                    total_ms=$(awk "BEGIN {printf \"%.2f\", $qkv_total + $attn_total + $dense_total + $mlp_total + $router_total + $experts_total}" 2>/dev/null || echo "0")
                fi
                
                # Write to CSV (k=0 for non-TopK)
                echo "$block_size,$batch_size,0,$INPUT_LENGTH,$OUTPUT_LENGTH,$qkv_avg,$qkv_count,$qkv_total,$attn_avg,$attn_count,$attn_total,$dense_avg,$dense_count,$dense_total,$mlp_avg,$mlp_count,$mlp_total,$router_avg,$router_count,$router_total,$experts_avg,$experts_count,$experts_total,$total_ms" >> "$OUTPUT_CSV"
                
                echo "  ✓ Success: qkv=${qkv_avg}ms, attention=${attn_avg}ms, dense=${dense_avg}ms, mlp=${mlp_avg}ms, router=${router_avg}ms, experts=${experts_avg}ms"
            else
                echo "  ✗ Failed: Check $temp_output for details"
                # Show last few lines of error
                echo "  Last 10 lines of output:"
                tail -n 10 "$temp_output" | sed 's/^/    /'
                echo "$block_size,$batch_size,0,$INPUT_LENGTH,$OUTPUT_LENGTH,ERROR,0,0,ERROR,0,0,ERROR,0,0,ERROR,0,0,ERROR,0,0,ERROR,0,0,ERROR" >> "$OUTPUT_CSV"
            fi
            
            # Cleanup
            rm -f "$temp_output"
            echo ""
        fi
    done
done

echo "=========================================="
echo "Scan complete!"
echo "Results CSV: $OUTPUT_CSV"
echo "Output Log: $OUTPUT_LOG"
echo "Error Log: $OUTPUT_ERR"
echo "=========================================="
