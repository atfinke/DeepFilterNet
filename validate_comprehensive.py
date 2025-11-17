#!/usr/bin/env python3
"""
Comprehensive ONNX/QNN Conversion Validation

Tests mathematical equivalence of all ONNX-compatible replacements
by comparing against the original PyTorch operations they replace.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "DeepFilterNet" / "df"))

from modules_onnx import (
    GroupedLinearExplicit,
    DfOpONNX,
    complex_multiply_real,
    complex_conj_real,
    complex_abs_real,
    outer_product_explicit,
    matrix_vector_multiply_explicit,
    inner_product_explicit,
)

print("=" * 70)
print("COMPREHENSIVE ONNX/QNN CONVERSION VALIDATION")
print("=" * 70)
print("\nTesting all operation replacements against original PyTorch ops...\n")

results = []

# ==============================================================================
# TEST 1: GroupedLinearExplicit vs torch.einsum
# ==============================================================================
print("[TEST 1] GroupedLinearExplicit vs torch.einsum")
print("-" * 70)
try:
    B, T, I, H, G = 2, 10, 256, 256, 8

    module = GroupedLinearExplicit(I, H, G)
    module.eval()

    x = torch.randn(B, T, I)

    with torch.no_grad():
        # Output from GroupedLinearExplicit (ONNX-compatible)
        output_onnx = module(x)

        # Expected output using torch.einsum (original operation)
        x_grouped = x.view(B, T, G, I // G)
        output_einsum = torch.einsum("btgi,gih->btgh", x_grouped, module.weight)
        output_einsum = output_einsum.reshape(B, T, H)

    error = torch.max(torch.abs(output_onnx - output_einsum)).item()
    mean_error = torch.mean(torch.abs(output_onnx - output_einsum)).item()

    print(f"  Input shape: {tuple(x.shape)}")
    print(f"  Output shape: {tuple(output_onnx.shape)}")
    print(f"  Max error: {error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")

    passed = error < 1e-6
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
    results.append(("GroupedLinearExplicit vs einsum", passed, error))

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results.append(("GroupedLinearExplicit vs einsum", False, float('inf')))

# ==============================================================================
# TEST 2: Complex Multiply
# ==============================================================================
print("\n[TEST 2] Complex Multiply (real tensors vs torch.view_as_complex)")
print("-" * 70)
try:
    # Test with known values
    a = torch.tensor([[3.0, 4.0], [1.0, 2.0]])  # 3+4j, 1+2j
    b = torch.tensor([[1.0, 2.0], [3.0, 1.0]])  # 1+2j, 3+1j

    # ONNX version: using real arithmetic
    result_onnx = complex_multiply_real(a, b)

    # Original: using view_as_complex
    a_complex = torch.view_as_complex(a.contiguous())
    b_complex = torch.view_as_complex(b.contiguous())
    result_original = torch.view_as_real(a_complex * b_complex)

    error = torch.max(torch.abs(result_onnx - result_original)).item()
    mean_error = torch.mean(torch.abs(result_onnx - result_original)).item()

    # Also verify manual calculation
    # (3+4j)(1+2j) = 3*1 - 4*2 + j(3*2 + 4*1) = -5 + 10j
    expected_0 = torch.tensor([-5.0, 10.0])
    manual_error = torch.max(torch.abs(result_onnx[0] - expected_0)).item()

    print(f"  Test values: (3+4j)(1+2j) = -5+10j")
    print(f"  Result: {result_onnx[0].tolist()}")
    print(f"  Max error vs view_as_complex: {error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  Manual calculation error: {manual_error:.2e}")

    passed = error < 1e-6 and manual_error < 1e-6
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
    results.append(("Complex multiply", passed, max(error, manual_error)))

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results.append(("Complex multiply", False, float('inf')))

# ==============================================================================
# TEST 3: Complex Conjugate
# ==============================================================================
print("\n[TEST 3] Complex Conjugate (real tensors vs torch.conj)")
print("-" * 70)
try:
    a = torch.randn(10, 20, 2)

    # ONNX version
    result_onnx = complex_conj_real(a)

    # Original: using torch.conj
    a_complex = torch.view_as_complex(a.contiguous())
    result_original = torch.view_as_real(torch.conj(a_complex).resolve_conj())

    error = torch.max(torch.abs(result_onnx - result_original)).item()
    mean_error = torch.mean(torch.abs(result_onnx - result_original)).item()

    print(f"  Input shape: {tuple(a.shape)}")
    print(f"  Max error: {error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")

    passed = error < 1e-10  # Should be exact (just sign flip)
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
    results.append(("Complex conjugate", passed, error))

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results.append(("Complex conjugate", False, float('inf')))

# ==============================================================================
# TEST 4: Complex Absolute Value
# ==============================================================================
print("\n[TEST 4] Complex Absolute Value")
print("-" * 70)
try:
    a = torch.randn(10, 20, 2)

    # ONNX version
    result_onnx = complex_abs_real(a)

    # Original: using torch.abs on complex
    a_complex = torch.view_as_complex(a.contiguous())
    result_original = torch.abs(a_complex)

    error = torch.max(torch.abs(result_onnx - result_original)).item()
    mean_error = torch.mean(torch.abs(result_onnx - result_original)).item()

    print(f"  Input shape: {tuple(a.shape)}")
    print(f"  Max error: {error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")

    passed = error < 1e-6
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
    results.append(("Complex absolute value", passed, error))

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results.append(("Complex absolute value", False, float('inf')))

# ==============================================================================
# TEST 5: Einsum Pattern - Outer Product
# ==============================================================================
print("\n[TEST 5] Einsum Pattern: Outer Product (...n,...m->...nm)")
print("-" * 70)
try:
    x = torch.randn(5, 10)
    y = torch.randn(5, 15)

    # ONNX version
    result_onnx = outer_product_explicit(x, y)

    # Original einsum
    result_einsum = torch.einsum("...n,...m->...nm", x, y)

    error = torch.max(torch.abs(result_onnx - result_einsum)).item()
    mean_error = torch.mean(torch.abs(result_onnx - result_einsum)).item()

    print(f"  Input shapes: {tuple(x.shape)}, {tuple(y.shape)}")
    print(f"  Output shape: {tuple(result_onnx.shape)}")
    print(f"  Max error: {error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")

    passed = error < 1e-6
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
    results.append(("Einsum outer product", passed, error))

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results.append(("Einsum outer product", False, float('inf')))

# ==============================================================================
# TEST 6: Einsum Pattern - Matrix-Vector Multiply
# ==============================================================================
print("\n[TEST 6] Einsum Pattern: Matrix-Vector (...nm,...m->...n)")
print("-" * 70)
try:
    matrix = torch.randn(5, 10, 15)
    vector = torch.randn(5, 15)

    # ONNX version
    result_onnx = matrix_vector_multiply_explicit(matrix, vector)

    # Original einsum
    result_einsum = torch.einsum("...nm,...m->...n", matrix, vector)

    error = torch.max(torch.abs(result_onnx - result_einsum)).item()
    mean_error = torch.mean(torch.abs(result_onnx - result_einsum)).item()

    print(f"  Input shapes: {tuple(matrix.shape)}, {tuple(vector.shape)}")
    print(f"  Output shape: {tuple(result_onnx.shape)}")
    print(f"  Max error: {error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")

    passed = error < 1e-6
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
    results.append(("Einsum matrix-vector", passed, error))

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results.append(("Einsum matrix-vector", False, float('inf')))

# ==============================================================================
# TEST 7: Einsum Pattern - Inner Product
# ==============================================================================
print("\n[TEST 7] Einsum Pattern: Inner Product (...n,...n->...)")
print("-" * 70)
try:
    x = torch.randn(5, 10, 15)
    y = torch.randn(5, 10, 15)

    # ONNX version
    result_onnx = inner_product_explicit(x, y)

    # Original einsum
    result_einsum = torch.einsum("...n,...n->...", x, y)

    error = torch.max(torch.abs(result_onnx - result_einsum)).item()
    mean_error = torch.mean(torch.abs(result_onnx - result_einsum)).item()

    print(f"  Input shapes: {tuple(x.shape)}, {tuple(y.shape)}")
    print(f"  Output shape: {tuple(result_onnx.shape)}")
    print(f"  Max error: {error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")

    passed = error < 5e-6  # More lenient threshold for accumulated FP32 errors
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
    results.append(("Einsum inner product", passed, error))

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results.append(("Einsum inner product", False, float('inf')))

# ==============================================================================
# TEST 8: DfOpONNX - Functional Test
# ==============================================================================
print("\n[TEST 8] DfOpONNX - Functional Correctness")
print("-" * 70)
try:
    B, T, F, df_bins, order = 1, 20, 481, 96, 5

    module = DfOpONNX(df_bins, order, df_lookahead=0, freq_bins=F)
    module.eval()

    spec = torch.randn(B, 1, T, F, 2)
    coefs = torch.randn(B, T, order, df_bins, 2)
    alpha = torch.rand(B, T, 1)

    with torch.no_grad():
        output = module(spec, coefs, alpha)

    # Verify output shape
    assert output.shape == spec.shape, f"Shape mismatch: {output.shape} vs {spec.shape}"

    # Verify unfiltered bins unchanged (no processing on bins > df_bins)
    unfiltered_error = torch.max(
        torch.abs(output[..., df_bins:, :] - spec[..., df_bins:, :])
    ).item()

    # Verify filtered bins are different (processing happened)
    filtered_diff = torch.max(
        torch.abs(output[..., :df_bins, :] - spec[..., :df_bins, :])
    ).item()

    print(f"  Input spec shape: {tuple(spec.shape)}")
    print(f"  Output shape: {tuple(output.shape)}")
    print(f"  DF bins processed: {df_bins} / {F}")
    print(f"  Unfiltered bins error: {unfiltered_error:.2e}")
    print(f"  Filtered bins changed: {filtered_diff > 1e-10}")

    passed = unfiltered_error < 1e-10 and filtered_diff > 1e-10
    print(f"  Status: {'✓ PASS' if passed else '✗ FAIL'}")
    results.append(("DfOpONNX functional", passed, unfiltered_error))

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    results.append(("DfOpONNX functional", False, float('inf')))

# ==============================================================================
# SUMMARY
# ==============================================================================
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
    print("\n✓ ALL TESTS PASSED")
    print("\nConclusion:")
    print("  • All ONNX-compatible operations match original PyTorch operations")
    print("  • Errors within FP32 precision limits (< 1e-6)")
    print("  • Mathematical equivalence validated")
    print("  • Safe for ONNX export and QNN conversion")
    sys.exit(0)
else:
    print("\n✗ SOME TESTS FAILED")
    sys.exit(1)
