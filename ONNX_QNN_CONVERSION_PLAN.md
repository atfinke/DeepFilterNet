# DeepFilterNet ONNX/QNN Conversion Plan

## Executive Summary

This document outlines the plan to convert DeepFilterNet models to ONNX format and subsequently to QNN for Hexagon NPU acceleration. The conversion requires replacing at least 8 PyTorch operations that are not well-supported in ONNX/QNN with compatible alternatives.

## Current Architecture Analysis

### Model Structure
- **DeepFilterNet** operates at 48kHz sampling rate
- Three main components: Encoder, ERB Decoder, DF Decoder
- No custom CUDA kernels (pure PyTorch implementation)
- Existing ONNX export functionality in `/DeepFilterNet/df/scripts/export.py`

### Key Finding
**No custom CUDA operators exist** in the codebase. However, several PyTorch operations used are problematic for ONNX/QNN deployment:

## Operations Requiring Replacement (≥8 Ops)

### 1. **torch.einsum** (HIGH PRIORITY)
- **Location**:
  - `modules.py:774` - GroupedLinearEinsum
  - `multiframe.py:99, 107, 123, 136, 309, 404-405, 457, 527-528, 612, 615, 617`
  - `loss.py:356-357`
- **Issue**: Limited ONNX opset support, not optimized for QNN/Hexagon
- **Solution**: Replace with explicit `torch.matmul`, `torch.bmm`, or `torch.einsum` decomposition
- **Example Patterns**:
  ```python
  # Pattern: "btgi,gih->btgh" (GroupedLinearEinsum)
  # Replace with: batch matrix multiply + reshape

  # Pattern: "...tfn,...ntf->...tf" (Deep Filter)
  # Replace with: transpose + batch matmul

  # Pattern: "...n,...m->...mn" (Outer product)
  # Replace with: unsqueeze + multiply
  ```

### 2. **torch.as_strided** (HIGH PRIORITY)
- **Location**: `modules.py:500` - Used in DfOp for sliding window operations
- **Issue**: Memory layout operations not portable to ONNX
- **Solution**: Replace with explicit padding + loop unrolling or unfold + reshape
- **Impact**: DfOp.forward_real_strided, DfOp.forward_complex_strided

### 3. **torch.view_as_complex / torch.view_as_real** (MEDIUM PRIORITY)
- **Location**: Throughout codebase for complex number handling
  - `modules.py:418-419` - DfOp.forward_complex_strided
  - `utils.py` - as_complex(), as_real() utility functions
- **Issue**: Complex number support varies by ONNX opset, QNN may not support
- **Solution**: Keep all operations in real tensor format [B, T, F, 2] for re/im
- **Note**: Already partially handled via feat_spec transpose in export.py:178

### 4. **torch.jit.script** (MEDIUM PRIORITY)
- **Location**:
  - `deepfilternet.py:246` - DfOp compilation
  - `export.py:99` - Export with jit=True
- **Issue**: TorchScript may introduce incompatible graph patterns
- **Solution**: Use direct PyTorch export without TorchScript compilation

### 5. **Tensor.unfold()** (MEDIUM PRIORITY)
- **Location**:
  - `modules.py:398` - DfOp.forward_real_unfold
  - `modules.py:852` - LocalSnrTarget._local_energy
- **Issue**: Unfold creates memory views that may not export cleanly
- **Solution**: Replace with explicit loops or reshape + gather operations

### 6. **Complex arithmetic operations** (MEDIUM PRIORITY)
- **Location**: `multiframe.py` - torch.conj(), complex multiplication/division
- **Issue**: QNN may not support complex dtypes
- **Solution**: Implement complex operations using real tensors:
  ```python
  # Complex multiply: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
  # Replace tensor operations with explicit real/imag computations
  ```

### 7. **GroupedGRU custom implementation** (LOW PRIORITY)
- **Location**: `modules.py:503-661` - GroupedGRULayer, GroupedGRU
- **Issue**: Custom GRU with manual splitting may not optimize well
- **Solution**:
  - Option A: Use standard nn.GRU with proper reshaping
  - Option B: Export grouped operations as separate GRU instances
- **Note**: May already work with current export (test first)

### 8. **Dynamic tensor shape operations** (LOW PRIORITY)
- **Location**: Various locations with `.shape`, `.size()` in forward pass
- **Issue**: Dynamic shapes can cause ONNX export failures
- **Solution**: Use fixed shapes or mark as dynamic axes in export

### 9. **torch.roll** (LOW PRIORITY)
- **Location**: `modules.py:454` - DfOp.forward_real_hidden_state_loop
- **Issue**: Roll operation may not be optimally supported
- **Solution**: Replace with concatenation + slicing

### 10. **Tensor.clone() / Tensor.detach()** (OPTIMIZATION)
- **Location**: Various locations
- **Issue**: May create unnecessary copies in ONNX graph
- **Solution**: Minimize usage where not strictly necessary

### 11. **F.interpolate** (OPTIMIZATION)
- **Location**: `modules.py:203` - FreqUpsample
- **Issue**: Interpolation modes may have limited QNN support
- **Solution**: Use only "nearest" mode, or implement as explicit operations

### 12. **Batch normalization with running stats** (QNN SPECIFIC)
- **Location**: Conv2dNormAct, ConvTranspose2dNormAct
- **Issue**: BN running stats need to be frozen for inference
- **Solution**: Ensure model is in eval mode, or fuse BN into conv weights

## Implementation Strategy

### Phase 1: Create ONNX-Compatible Module Library
Create `modules_onnx.py` with compatible replacements:

1. **GroupedLinearExplicit** - Replace GroupedLinearEinsum
   ```python
   class GroupedLinearExplicit(nn.Module):
       # Use reshape + batch matmul instead of einsum
       def forward(self, x):
           # Reshape to [B*T, G, I/G]
           # Matmul with weight [G, I/G, H/G]
           # Reshape back to [B, T, H]
   ```

2. **DfOpONNX** - Simplified deep filter without as_strided
   ```python
   class DfOpONNX(nn.Module):
       # Use forward_real_loop or forward_real_unfold
       # Avoid torch.as_strided and torch.view_as_complex
   ```

3. **ComplexMultiplyReal** - Complex ops using real tensors
   ```python
   def complex_multiply_real(a, b):
       # a, b shape: [..., 2] where last dim is [real, imag]
       real = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
       imag = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
       return torch.stack([real, imag], dim=-1)
   ```

### Phase 2: Update Model Definitions
Modify `deepfilternet2.py` and `deepfilternet3.py`:
- Add `onnx_export` flag to model constructor
- Conditionally use ONNX-compatible modules when flag is set
- Maintain backward compatibility with original implementation

### Phase 3: Enhanced ONNX Export
Update `export.py`:
- Set models to use ONNX-compatible operations
- Remove TorchScript compilation (jit=False for all exports)
- Increase opset version to 15+ for better operation coverage
- Add comprehensive validation suite

### Phase 4: QNN Conversion Pipeline
Create `scripts/convert_to_qnn.py`:

#### 4.1 ONNX to QNN Conversion
```bash
# Use Qualcomm QNN SDK tools
qnn-onnx-converter \
    --input_network enc.onnx \
    --output_path enc.dlc \
    --input_dim feat_erb "1,1,100,32" feat_spec "1,2,100,96"
```

#### 4.2 Quantization Strategy
Options:
1. **Post-Training Quantization (PTQ)** - Faster, good baseline
   ```bash
   qnn-quantizer \
       --input_dlc enc.dlc \
       --output_dlc enc_quantized.dlc \
       --input_list calibration_inputs.txt
   ```

2. **Quantization-Aware Training (QAT)** - Better accuracy
   - Use PyTorch quantization API
   - Simulate quantization during fine-tuning
   - Export quantized model to ONNX

#### 4.3 Custom Operators (if needed)
If QNN doesn't support certain operations:
- Implement custom DSP/HVX kernels following QNN specification
- Register custom operators in QNN runtime
- Potential candidates: Complex operations, specific einsum patterns

### Phase 5: Validation & Optimization
1. **Accuracy Validation**:
   - Run inference on test set with PyTorch, ONNX Runtime, QNN
   - Compare outputs at each stage (tolerance: rtol=1e-3, atol=1e-4)
   - Measure SNR improvement, PESQ, STOI metrics

2. **Performance Benchmarking**:
   - Measure latency on Hexagon NPU
   - Profile bottleneck operations
   - Optimize quantization bit-widths (INT8, INT16)

3. **Iterative Refinement**:
   - If accuracy drops: adjust quantization, try QAT
   - If performance is poor: implement critical ops as custom QNN ops
   - Balance accuracy vs latency trade-offs

## File Structure

```
DeepFilterNet/
├── DeepFilterNet/df/
│   ├── modules_onnx.py          # NEW: ONNX-compatible modules
│   ├── deepfilternet2_onnx.py   # NEW: ONNX model variant
│   ├── deepfilternet3_onnx.py   # NEW: ONNX model variant
│   └── scripts/
│       ├── export_onnx.py       # UPDATED: Enhanced export
│       ├── validate_onnx.py     # NEW: Validation suite
│       ├── convert_to_qnn.py    # NEW: QNN conversion
│       └── benchmark_qnn.py     # NEW: QNN benchmarking
└── models/
    └── qnn/                      # NEW: QNN model artifacts
        ├── enc.dlc
        ├── erb_dec.dlc
        ├── df_dec.dlc
        └── quantization/
            └── calibration_data/
```

## Success Criteria

### Phase 1 (ONNX Export) - Complete when:
- [ ] All 3 models (enc, erb_dec, df_dec) export to ONNX without errors
- [ ] ONNX Runtime can execute models successfully
- [ ] Output accuracy vs PyTorch: RMSE < 1e-3
- [ ] At least 8 operations successfully replaced with ONNX-compatible versions

### Phase 2 (QNN Conversion) - Complete when:
- [ ] All ONNX models convert to QNN .dlc format
- [ ] QNN Runtime can execute on Hexagon NPU simulator
- [ ] Custom operators (if needed) pass QNN validation

### Phase 3 (Optimization) - Complete when:
- [ ] Quantized models maintain > 95% of FP32 accuracy
- [ ] Inference latency < 50ms per frame on target NPU
- [ ] Memory footprint < 100MB

## Timeline Estimate

- Phase 1 (ONNX): 3-4 days
- Phase 2 (QNN): 2-3 days
- Phase 3 (Optimization): 3-5 days
- **Total**: ~10 days with testing and iteration

## Dependencies

### Software Requirements
- PyTorch >= 1.13
- ONNX >= 1.14
- ONNX Runtime >= 1.15
- onnxsim
- Qualcomm QNN SDK >= 2.x
- numpy, scipy (for metrics)

### Hardware Requirements
- Hexagon NPU simulator (for testing)
- Target device with Snapdragon SoC (for final validation)

## Risk Mitigation

1. **Risk**: Accuracy degradation in quantization
   - **Mitigation**: Start with PTQ, fall back to QAT if needed

2. **Risk**: Unsupported QNN operations
   - **Mitigation**: Implement custom operators, or simplify model architecture

3. **Risk**: Performance not meeting targets
   - **Mitigation**: Profile and optimize critical paths, use mixed precision

## Next Steps

1. Create `modules_onnx.py` with ONNX-compatible implementations
2. Test each operation replacement individually
3. Update model definitions with ONNX mode
4. Run export and validation
5. Begin QNN conversion

---

**Document Version**: 1.0
**Date**: 2025-11-17
**Author**: Claude Code
