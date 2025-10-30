#!/usr/bin/env python3
"""
Utility to check if an ONNX model uses Div operations.
Returns exit code 0 if NO Div, exit code 1 if Div found.
"""

import sys
import onnx

def check_model_for_div(model_path):
    """Check if model contains Div operations"""
    try:
        model = onnx.load(model_path)

        # Check for Div operations
        div_nodes = [node for node in model.graph.node if node.op_type == 'Div']

        if div_nodes:
            print(f"WARNING: Model contains {len(div_nodes)} Div operation(s)")
            print("Division operations may cause verification failures in JOLT Atlas.")
            print("Consider using a division-free version of this model.")
            for i, node in enumerate(div_nodes):
                print(f"  - Div operation {i+1}: {node.name}")
            return 1  # Div found
        else:
            return 0  # No Div

    except Exception as e:
        print(f"ERROR checking model: {e}")
        return 2  # Error

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: check_model_div.py <model.onnx>")
        sys.exit(2)

    model_path = sys.argv[1]
    exit_code = check_model_for_div(model_path)
    sys.exit(exit_code)
