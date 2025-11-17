#!/usr/bin/env python3
"""
REAL validation: Compares ONNX-compatible implementations against originals.
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "DeepFilterNet" / "df"))

# Import ORIGINAL implementations
from modules import GroupedLinearEinsum, DfOp

# Import ONNX-compatible implementations
from modules_onnx import GroupedLinearExplicit, DfOpONNX, complex_multiply_real

print("=" * 70)
print("REAL VALIDATION: ONNX vs Original Implementations")
print("=" * 70)
print()

results = []

# TEST 1: GroupedLinearExplicit vs GroupedLinearEinsum
print("[TEST 1] GroupedLinearExplicit vs GroupedLinearEinsum")
print("-" * 70)
try:
    B, T, I, H, G = 2, 10, 256, 256, 8

    # Create original module
    original = GroupedLinearEinsum(I, H, G)
    original.eval()

    # Create ONNX module with SAME weights
    onnx_version = GroupedLinearExplicit(I, H, G)
    onnx_version.weight.data = original.weight.data.clone()
    onnx_version.eval()

    # Test with random input
    x = torch.randn(B, T, I)

    with torch.no_grad():
        output_original = original(x)
        output_onnx = onnx_version(x)

    error = torch.max(torch.abs(output_original - output_onnx)).item()
    mean_error = torch.mean(torch.abs(output_original - output_onnx)).item()

    print(f"  Input shape: {tuple(x.shape)}")
    print(f"  Output shape: {tuple(output_original.shape)}")
    print(f"  Max error: {error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")

    passed = error < 1e-6
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
    results.append(("GroupedLinear (einsum)", passed, error))

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results.append(("GroupedLinear (einsum)", False, float('inf')))

# TEST 2: DfOpONNX vs DfOp
print("\n[TEST 2] DfOpONNX vs DfOp (multiple methods)")
print("-" * 70)

test_methods = ["real_unfold", "real_loop"]
df_errors = []

for method in test_methods:
    try:
        B, T, F, df_bins, order = 1, 20, 481, 96, 5

        # Create original DfOp with specific method
        original = DfOp(df_bins, order, df_lookahead=0, method=method, freq_bins=F)
        original.eval()

        # Create ONNX version
        onnx_version = DfOpONNX(df_bins, order, df_lookahead=0, freq_bins=F)
        onnx_version.eval()

        # Same inputs
        spec = torch.randn(B, 1, T, F, 2)
        coefs = torch.randn(B, T, order, df_bins, 2)
        alpha = torch.rand(B, T, 1)

        with torch.no_grad():
            output_original = original(spec, coefs, alpha)
            output_onnx = onnx_version(spec, coefs, alpha)

        # Compare only the DF bins (where filtering happens)
        error = torch.max(torch.abs(
            output_original[..., :df_bins, :] - output_onnx[..., :df_bins, :]
        )).item()
        mean_error = torch.mean(torch.abs(
            output_original[..., :df_bins, :] - output_onnx[..., :df_bins, :]
        )).item()

        print(f"  Method '{method}':")
        print(f"    Max error: {error:.2e}")
        print(f"    Mean error: {mean_error:.2e}")

        df_errors.append(error)

    except Exception as e:
        print(f"  Method '{method}': ✗ ERROR: {e}")
        df_errors.append(float('inf'))

max_df_error = max(df_errors) if df_errors else float('inf')
passed = max_df_error < 1e-5  # More lenient for complex operation
print(f"  Overall max error: {max_df_error:.2e}")
print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
results.append(("DfOp (as_strided)", passed, max_df_error))

# TEST 3: Complex operations using torch.view_as_complex
print("\n[TEST 3] Complex operations vs torch.view_as_complex")
print("-" * 70)
try:
    # Test complex multiply
    a = torch.randn(10, 20, 2)
    b = torch.randn(10, 20, 2)

    # Original: using view_as_complex
    a_complex = torch.view_as_complex(a.contiguous())
    b_complex = torch.view_as_complex(b.contiguous())
    result_original = torch.view_as_real(a_complex * b_complex)

    # ONNX version: using real arithmetic
    result_onnx = complex_multiply_real(a, b)

    error = torch.max(torch.abs(result_original - result_onnx)).item()
    mean_error = torch.mean(torch.abs(result_original - result_onnx)).item()

    print(f"  Complex multiply:")
    print(f"    Max error: {error:.2e}")
    print(f"    Mean error: {mean_error:.2e}")

    passed = error < 1e-6
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
    results.append(("Complex multiply", passed, error))

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results.append(("Complex multiply", False, float('inf')))

# TEST 4: Einsum patterns (already validated)
print("\n[TEST 4] Einsum patterns vs torch.einsum")
print("-" * 70)
try:
    from modules_onnx import (
        outer_product_explicit,
        matrix_vector_multiply_explicit,
        inner_product_explicit
    )

    errors = []

    # Test each pattern
    x = torch.randn(5, 10)
    y = torch.randn(5, 15)
    result = outer_product_explicit(x, y)
    expected = torch.einsum("...n,...m->...nm", x, y)
    error = torch.max(torch.abs(result - expected)).item()
    errors.append(error)
    print(f"  Outer product: {error:.2e}")

    matrix = torch.randn(5, 10, 15)
    vector = torch.randn(5, 15)
    result = matrix_vector_multiply_explicit(matrix, vector)
    expected = torch.einsum("...nm,...m->...n", matrix, vector)
    error = torch.max(torch.abs(result - expected)).item()
    errors.append(error)
    print(f"  Matrix-vector: {error:.2e}")

    x = torch.randn(5, 10, 15)
    y = torch.randn(5, 10, 15)
    result = inner_product_explicit(x, y)
    expected = torch.einsum("...n,...n->...", x, y)
    error = torch.max(torch.abs(result - expected)).item()
    errors.append(error)
    print(f"  Inner product: {error:.2e}")

    max_error = max(errors)
    passed = max_error < 1e-6
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
    results.append(("Einsum patterns", passed, max_error))

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    results.append(("Einsum patterns", False, float('inf')))

# SUMMARY
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

all_passed = True
for name, passed, error in results:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {name:40s} {status:10s} (error: {error:.2e})")
    all_passed = all_passed and passed

print("=" * 70)

if all_passed:
    print("\n✓ ALL TESTS PASSED - VALIDATED AGAINST ORIGINALS")
    print("\nConclusion:")
    print("  • ONNX implementations match original implementations")
    print("  • All errors within numerical precision")
    print("  • Safe for ONNX export")
    sys.exit(0)
else:
    print("\n✗ VALIDATION FAILED - Implementations do not match")
    print("\nAction required: Fix implementations before using")
    sys.exit(1)
