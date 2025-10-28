#!/usr/bin/env python3
"""
Test all 10 curated ONNX models with comprehensive test cases.
Validates model execution and documents input/output signatures.
"""

import onnxruntime as ort
import numpy as np
import json
from pathlib import Path

# Test cases for each model based on CATALOG.md
TEST_CASES = {
    "simple_threshold": [
        {"inputs": {"amount": 5000, "balance": 10000}, "expected": 1, "desc": "Sufficient balance"},
        {"inputs": {"amount": 15000, "balance": 10000}, "expected": 0, "desc": "Insufficient balance"},
        {"inputs": {"amount": 10000, "balance": 10000}, "expected": 0, "desc": "Exact balance (edge case)"},
    ],
    "percentage_limit": [
        {"inputs": {"amount": 5000, "balance": 100000, "max_percentage": 10}, "expected": 1, "desc": "5% of balance (within 10% limit)"},
        {"inputs": {"amount": 15000, "balance": 100000, "max_percentage": 10}, "expected": 0, "desc": "15% of balance (exceeds 10% limit)"},
        {"inputs": {"amount": 9999, "balance": 100000, "max_percentage": 10}, "expected": 1, "desc": "Just under 10% limit"},
    ],
    "vendor_trust": [
        {"inputs": {"vendor_trust": 75, "min_trust": 50}, "expected": 1, "desc": "High trust vendor"},
        {"inputs": {"vendor_trust": 30, "min_trust": 50}, "expected": 0, "desc": "Low trust vendor"},
        {"inputs": {"vendor_trust": 50, "min_trust": 50}, "expected": 1, "desc": "Exactly minimum trust"},
    ],
    "velocity_1h": [
        {"inputs": {"amount": 5000, "spent_1h": 10000, "limit_1h": 20000}, "expected": 1, "desc": "Within hourly limit"},
        {"inputs": {"amount": 15000, "spent_1h": 10000, "limit_1h": 20000}, "expected": 0, "desc": "Exceeds hourly limit"},
        {"inputs": {"amount": 10000, "spent_1h": 10000, "limit_1h": 20000}, "expected": 1, "desc": "Exactly at limit"},
    ],
    "velocity_24h": [
        {"inputs": {"amount": 5000, "spent_24h": 20000, "limit_24h": 50000}, "expected": 1, "desc": "Within daily limit"},
        {"inputs": {"amount": 40000, "spent_24h": 20000, "limit_24h": 50000}, "expected": 0, "desc": "Exceeds daily limit"},
        {"inputs": {"amount": 30000, "spent_24h": 20000, "limit_24h": 50000}, "expected": 1, "desc": "Exactly at limit"},
    ],
    "daily_limit": [
        {"inputs": {"amount": 10000, "daily_spent": 5000, "daily_cap": 20000}, "expected": 1, "desc": "Within daily cap"},
        {"inputs": {"amount": 20000, "daily_spent": 5000, "daily_cap": 20000}, "expected": 0, "desc": "Exceeds daily cap"},
        {"inputs": {"amount": 15000, "daily_spent": 5000, "daily_cap": 20000}, "expected": 1, "desc": "Exactly at cap"},
    ],
    "age_gate": [
        {"inputs": {"age": 25, "min_age": 21}, "expected": 1, "desc": "Adult over 21"},
        {"inputs": {"age": 18, "min_age": 21}, "expected": 0, "desc": "Under age limit"},
        {"inputs": {"age": 21, "min_age": 21}, "expected": 1, "desc": "Exactly minimum age"},
    ],
    "multi_factor": [
        {"inputs": {"amount": 5000, "balance": 100000, "spent_24h": 20000, "limit_24h": 50000, "vendor_trust": 75, "min_trust": 50}, "expected": 1, "desc": "All checks pass"},
        {"inputs": {"amount": 5000, "balance": 3000, "spent_24h": 20000, "limit_24h": 50000, "vendor_trust": 75, "min_trust": 50}, "expected": 0, "desc": "Insufficient balance"},
        {"inputs": {"amount": 5000, "balance": 100000, "spent_24h": 48000, "limit_24h": 50000, "vendor_trust": 75, "min_trust": 50}, "expected": 0, "desc": "Velocity limit exceeded"},
        {"inputs": {"amount": 5000, "balance": 100000, "spent_24h": 20000, "limit_24h": 50000, "vendor_trust": 30, "min_trust": 50}, "expected": 0, "desc": "Low vendor trust"},
    ],
    "composite_scoring": [
        {"inputs": {"amount": 5000, "balance": 100000, "vendor_trust": 75, "user_history": 80}, "expected": 1, "desc": "High composite score"},
        {"inputs": {"amount": 5000, "balance": 6000, "vendor_trust": 20, "user_history": 30}, "expected": 0, "desc": "Low composite score"},
    ],
    "risk_neural": [
        {"inputs": {"amount": 5000, "balance": 100000, "velocity_1h": 5000, "velocity_24h": 20000, "vendor_trust": 75}, "expected": 1, "desc": "Low risk transaction"},
        {"inputs": {"amount": 50000, "balance": 60000, "velocity_1h": 15000, "velocity_24h": 80000, "vendor_trust": 30}, "expected": 0, "desc": "High risk transaction"},
    ],
}

def test_model(model_path: Path):
    """Test a single ONNX model with its test cases."""
    model_name = model_path.stem

    print(f"\n{'='*70}")
    print(f"Testing: {model_name}.onnx")
    print(f"{'='*70}")

    try:
        # Load model
        session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])

        # Get input/output info
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]

        print(f"\nInputs:  {input_names}")
        print(f"Outputs: {output_names}")

        # Get test cases
        test_cases = TEST_CASES.get(model_name, [])
        if not test_cases:
            print(f"⚠️  No test cases defined for {model_name}")
            return False

        print(f"\nRunning {len(test_cases)} test cases...")

        all_passed = True
        for i, test_case in enumerate(test_cases, 1):
            inputs = test_case["inputs"]
            expected = test_case["expected"]
            desc = test_case["desc"]

            # Prepare inputs as numpy arrays
            input_dict = {name: np.array([inputs[name]], dtype=np.int32) for name in input_names}

            # Run inference
            outputs = session.run(output_names, input_dict)

            # Check result - use 'approved' output if present, otherwise use first output
            if 'approved' in output_names:
                approved_idx = output_names.index('approved')
                result = outputs[approved_idx][0]
            else:
                result = outputs[0][0]  # First output, first element

            passed = (result == expected)

            status = "✅" if passed else "❌"
            print(f"  {status} Test {i}: {desc}")
            print(f"     Inputs: {inputs}")

            # Show all outputs for multi-output models
            if len(output_names) > 1:
                output_strs = [f"{name}={outputs[i][0]}" for i, name in enumerate(output_names)]
                print(f"     Outputs: {', '.join(output_strs)}")
                print(f"     Expected approved: {expected}, Got: {result}")
            else:
                print(f"     Expected: {expected}, Got: {result}")

            if not passed:
                all_passed = False

        if all_passed:
            print(f"\n✅ {model_name}: ALL TESTS PASSED")
        else:
            print(f"\n❌ {model_name}: SOME TESTS FAILED")

        return all_passed

    except Exception as e:
        print(f"\n❌ ERROR testing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all curated models."""
    curated_dir = Path(__file__).parent

    print("="*70)
    print("Testing All Curated ONNX Models")
    print("="*70)

    # Find all .onnx files
    onnx_files = sorted(curated_dir.glob("*.onnx"))

    if not onnx_files:
        print("❌ No ONNX files found in curated directory")
        return

    print(f"\nFound {len(onnx_files)} models to test\n")

    results = {}
    for model_path in onnx_files:
        passed = test_model(model_path)
        results[model_path.stem] = passed

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed_count = sum(results.values())
    total_count = len(results)

    for model_name, passed in sorted(results.items()):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {model_name}")

    print(f"\n{'='*70}")
    print(f"Results: {passed_count}/{total_count} models passed all tests")
    print(f"{'='*70}")

    # Write results to JSON
    results_file = curated_dir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": "2025-10-28",
            "total_models": total_count,
            "passed": passed_count,
            "failed": total_count - passed_count,
            "results": {k: "PASS" if v else "FAIL" for k, v in results.items()}
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return passed_count == total_count

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
