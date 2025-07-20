#!/usr/bin/env python3
"""
LPC prediction wrapper script that calls the C helper program.

This script:
1. Reads LPC coefficients and warmup samples from binary files
2. Calls the C LPC prediction helper
3. Outputs predicted samples to stdout
"""

import sys
import os
import subprocess
import tempfile

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 lpc_predict.py <coefficients_file> <warmup_file> <n_predicted_samples>", file=sys.stderr)
        print("  coefficients_file: binary file containing float32 LPC coefficients", file=sys.stderr)
        print("  warmup_file: binary file containing float32 warmup samples", file=sys.stderr)
        print("  n_predicted_samples: number of samples to predict", file=sys.stderr)
        sys.exit(1)
    
    coefficients_filename = sys.argv[1]
    warmup_filename = sys.argv[2]
    n_predicted_samples = sys.argv[3]
    
    # Validate n_predicted_samples
    try:
        n_predicted_samples_int = int(n_predicted_samples)
        if n_predicted_samples_int <= 0:
            raise ValueError("n_predicted_samples must be positive")
    except ValueError:
        print(f"Error: Invalid n_predicted_samples: {n_predicted_samples}", file=sys.stderr)
        sys.exit(1)
    
    # Path to C helper program
    script_dir = os.path.dirname(os.path.abspath(__file__))
    helper_path = os.path.join(script_dir, "lpc_predict_helper")
    
    # Check if C helper exists
    if not os.path.exists(helper_path):
        print(f"Error: C helper not found at {helper_path}", file=sys.stderr)
        print("Please run 'make' in the naive_lpc_helpers directory to compile the helpers", file=sys.stderr)
        sys.exit(1)
    
    # Check if input files exist
    if not os.path.exists(coefficients_filename):
        print(f"Error: Coefficients file not found: {coefficients_filename}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(warmup_filename):
        print(f"Error: Warmup file not found: {warmup_filename}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Call C helper
        result = subprocess.run(
            [helper_path, coefficients_filename, warmup_filename, n_predicted_samples],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        # Output predicted samples to stdout
        sys.stdout.buffer.write(result.stdout)
        
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode("utf-8", errors="ignore") if e.stderr else "No error message"
        print(f"C LPC helper failed (exit {e.returncode}): {stderr_text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error calling C LPC helper: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 