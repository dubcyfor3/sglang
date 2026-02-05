#!/usr/bin/env python3
"""Helper script to parse timing statistics from benchmark output."""
import sys
import re
import json

def parse_timing_stats(output_text):
    """Parse timing statistics from benchmark output."""
    stats = {}
    
    # Look for the Per-Component Timing Statistics section
    in_section = False
    found_header = False
    
    for line in output_text.split('\n'):
        # Check if we're entering the stats section
        if "Per-Component Timing Statistics" in line:
            in_section = True
            continue
        
        # Skip until we find the header
        if in_section and "Component" in line and "Avg (ms)" in line:
            found_header = True
            continue
        
        # Skip separator lines
        if found_header and "---" in line:
            continue
        
        # Check if we're leaving the stats section
        if in_section and found_header and "====" in line:
            break
        
        # Parse component lines
        if in_section and found_header and line.strip():
            # Format: attention                               12.34        100        1234.00
            # Use regex to match component name and numbers
            match = re.match(r'^(\w+)\s+([0-9]+\.[0-9]+)\s+([0-9]+)\s+([0-9]+\.[0-9]+)', line.strip())
            if match:
                component = match.group(1)
                avg_ms = float(match.group(2))
                count = int(match.group(3))
                total_ms = float(match.group(4))
                
                # Extract all component types: qkv, attention, dense, mlp, router, experts
                if component in ['qkv', 'attention', 'dense', 'mlp', 'router', 'experts']:
                    stats[component] = {
                        'avg': avg_ms,
                        'count': count,
                        'total': total_ms
                    }
    
    return stats

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: parse_timing_stats.py <output_file>")
        sys.exit(1)
    
    output_file = sys.argv[1]
    try:
        with open(output_file, 'r') as f:
            output_text = f.read()
        
        stats = parse_timing_stats(output_text)
        print(json.dumps(stats, indent=2))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
