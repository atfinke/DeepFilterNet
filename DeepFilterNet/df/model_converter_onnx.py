"""
Model Converter for ONNX Export

This module provides utilities to convert DeepFilterNet models to ONNX-compatible versions
by replacing problematic operations with ONNX/QNN-friendly alternatives.
"""

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from loguru import logger

from df.modules import DfOp, GroupedLinearEinsum, SqueezedGRU, SqueezedGRU_S
from df.modules_onnx import DfOpONNX, GroupedLinearExplicit, SqueezedGRU_ONNX


def convert_model_to_onnx_compatible(model: nn.Module, inplace: bool = False) -> nn.Module:
    """
    Convert a DeepFilterNet model to ONNX-compatible version.

    This function recursively replaces all problematic operations:
    1. GroupedLinearEinsum -> GroupedLinearExplicit (removes torch.einsum)
    2. DfOp -> DfOpONNX (removes torch.as_strided and torch.view_as_complex)
    3. SqueezedGRU -> SqueezedGRU_ONNX (uses GroupedLinearExplicit)

    Args:
        model: PyTorch model to convert
        inplace: If True, modify model in place. Otherwise, create a copy.

    Returns:
        ONNX-compatible model with same weights but compatible operations
    """
    if not inplace:
        model = deepcopy(model)

    model = model.cpu()  # Ensure on CPU for export
    model.eval()  # Set to eval mode

    _recursive_module_replacement(model)

    logger.info("Model converted to ONNX-compatible version")
    return model


def _recursive_module_replacement(model: nn.Module, parent_name: str = ""):
    """
    Recursively replace incompatible modules with ONNX-compatible versions.

    Args:
        model: Module to process
        parent_name: Name of parent module (for logging)
    """
    modules_to_replace = []

    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name

        if isinstance(module, GroupedLinearEinsum):
            logger.info(f"Replacing GroupedLinearEinsum at {full_name}")
            new_module = _convert_grouped_linear_einsum(module)
            modules_to_replace.append((name, new_module))

        elif isinstance(module, DfOp):
            logger.info(f"Replacing DfOp at {full_name}")
            new_module = _convert_dfop(module)
            modules_to_replace.append((name, new_module))

        elif isinstance(module, (SqueezedGRU, SqueezedGRU_S)):
            logger.info(f"Replacing {module.__class__.__name__} at {full_name}")
            new_module = _convert_squeezed_gru(module)
            modules_to_replace.append((name, new_module))

        else:
            # Recursively process child modules
            _recursive_module_replacement(module, full_name)

    # Replace modules
    for name, new_module in modules_to_replace:
        setattr(model, name, new_module)


def _convert_grouped_linear_einsum(module: GroupedLinearEinsum) -> GroupedLinearExplicit:
    """
    Convert GroupedLinearEinsum to GroupedLinearExplicit.

    REPLACES: torch.einsum("btgi,gih->btgh") with explicit matmul operations
    """
    new_module = GroupedLinearExplicit(
        input_size=module.input_size,
        hidden_size=module.hidden_size,
        groups=module.groups
    )

    # Copy weights
    with torch.no_grad():
        new_module.weight.data.copy_(module.weight.data)

    logger.debug(f"  Copied weights shape: {new_module.weight.shape}")
    return new_module


def _convert_dfop(module: DfOp) -> DfOpONNX:
    """
    Convert DfOp to DfOpONNX.

    REPLACES:
    - torch.as_strided with explicit padding + slicing
    - torch.view_as_complex with real tensor operations
    - torch.jit.script compilation
    """
    new_module = DfOpONNX(
        df_bins=module.df_bins,
        df_order=module.df_order,
        df_lookahead=module.df_lookahead,
        freq_bins=module.freq_bins
    )

    logger.debug(f"  DfOp config: bins={module.df_bins}, order={module.df_order}, "
                 f"lookahead={module.df_lookahead}")
    return new_module


def _convert_squeezed_gru(module: nn.Module) -> SqueezedGRU_ONNX:
    """
    Convert SqueezedGRU or SqueezedGRU_S to SqueezedGRU_ONNX.

    REPLACES: GroupedLinearEinsum in linear layers with GroupedLinearExplicit
    """
    # Determine output size
    output_size = None
    if hasattr(module, 'linear_out') and not isinstance(module.linear_out, nn.Identity):
        if isinstance(module.linear_out, nn.Sequential) and len(module.linear_out) > 0:
            linear_out_module = module.linear_out[0]
            if hasattr(linear_out_module, 'hidden_size'):
                output_size = linear_out_module.hidden_size

    # Get linear_groups from input linear layer
    linear_groups = 8  # default
    if hasattr(module, 'linear_in') and isinstance(module.linear_in, nn.Sequential):
        if len(module.linear_in) > 0:
            linear_in_module = module.linear_in[0]
            if hasattr(linear_in_module, 'groups'):
                linear_groups = linear_in_module.groups

    # Get num_layers from GRU
    num_layers = 1
    if hasattr(module, 'gru') and hasattr(module.gru, 'num_layers'):
        num_layers = module.gru.num_layers

    new_module = SqueezedGRU_ONNX(
        input_size=module.input_size,
        hidden_size=module.hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        linear_groups=linear_groups,
        batch_first=True
    )

    # Copy weights from linear_in
    with torch.no_grad():
        if hasattr(module, 'linear_in') and isinstance(module.linear_in, nn.Sequential):
            if len(module.linear_in) > 0:
                old_linear = module.linear_in[0]
                if isinstance(old_linear, GroupedLinearEinsum):
                    new_module.linear_in.weight.data.copy_(old_linear.weight.data)
                    logger.debug(f"  Copied linear_in weights")

        # Copy GRU weights
        if hasattr(module, 'gru'):
            new_module.gru.load_state_dict(module.gru.state_dict())
            logger.debug(f"  Copied GRU weights")

        # Copy weights from linear_out
        if hasattr(module, 'linear_out') and isinstance(module.linear_out, nn.Sequential):
            if len(module.linear_out) > 0:
                old_linear_out = module.linear_out[0]
                if isinstance(old_linear_out, GroupedLinearEinsum):
                    if isinstance(new_module.linear_out, GroupedLinearExplicit):
                        new_module.linear_out.weight.data.copy_(old_linear_out.weight.data)
                        logger.debug(f"  Copied linear_out weights")

    return new_module


def validate_conversion(original_model: nn.Module, converted_model: nn.Module,
                       test_inputs: dict, rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    """
    Validate that converted model produces similar outputs to original.

    Args:
        original_model: Original PyTorch model
        converted_model: ONNX-compatible converted model
        test_inputs: Dictionary of test inputs
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if outputs match within tolerance, False otherwise
    """
    import numpy as np

    original_model.eval()
    converted_model.eval()

    with torch.no_grad():
        # Get outputs from both models
        if isinstance(test_inputs, dict):
            original_outputs = original_model(**test_inputs)
        else:
            original_outputs = original_model(*test_inputs)

        if isinstance(test_inputs, dict):
            converted_outputs = converted_model(**test_inputs)
        else:
            converted_outputs = converted_model(*test_inputs)

        # Handle tuple outputs
        if isinstance(original_outputs, tuple):
            all_close = True
            for i, (orig, conv) in enumerate(zip(original_outputs, converted_outputs)):
                try:
                    np.testing.assert_allclose(
                        orig.cpu().numpy(),
                        conv.cpu().numpy(),
                        rtol=rtol,
                        atol=atol
                    )
                    logger.info(f"  Output {i}: ✓ Match (max diff: "
                               f"{torch.max(torch.abs(orig - conv)).item():.2e})")
                except AssertionError as e:
                    logger.warning(f"  Output {i}: ✗ Mismatch - {e}")
                    all_close = False
            return all_close
        else:
            try:
                np.testing.assert_allclose(
                    original_outputs.cpu().numpy(),
                    converted_outputs.cpu().numpy(),
                    rtol=rtol,
                    atol=atol
                )
                logger.info(f"  Output: ✓ Match (max diff: "
                           f"{torch.max(torch.abs(original_outputs - converted_outputs)).item():.2e})")
                return True
            except AssertionError as e:
                logger.warning(f"  Output: ✗ Mismatch - {e}")
                return False


def count_replaced_operations(model: nn.Module) -> dict:
    """
    Count number of operations replaced in the model.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with counts of each operation type
    """
    counts = {
        'GroupedLinearExplicit': 0,
        'DfOpONNX': 0,
        'SqueezedGRU_ONNX': 0,
        'Total': 0
    }

    def _count_modules(m):
        if isinstance(m, GroupedLinearExplicit):
            counts['GroupedLinearExplicit'] += 1
            counts['Total'] += 1
        elif isinstance(m, DfOpONNX):
            counts['DfOpONNX'] += 1
            counts['Total'] += 1
        elif isinstance(m, SqueezedGRU_ONNX):
            counts['SqueezedGRU_ONNX'] += 1
            # Each SqueezedGRU_ONNX contains 2 GroupedLinearExplicit
            counts['Total'] += 2

    model.apply(_count_modules)

    return counts


def print_conversion_summary(counts: dict):
    """Print summary of conversions."""
    logger.info("=" * 60)
    logger.info("ONNX Conversion Summary")
    logger.info("=" * 60)
    logger.info(f"GroupedLinearExplicit (replaces einsum):     {counts['GroupedLinearExplicit']}")
    logger.info(f"DfOpONNX (replaces as_strided + complex):    {counts['DfOpONNX']}")
    logger.info(f"SqueezedGRU_ONNX (uses explicit ops):        {counts['SqueezedGRU_ONNX']}")
    logger.info("-" * 60)
    logger.info(f"Total operations replaced:                   {counts['Total']}")
    logger.info("=" * 60)
    logger.info("\nOperations replaced include:")
    logger.info("  1. torch.einsum -> explicit matmul/bmm")
    logger.info("  2. torch.as_strided -> explicit padding + slicing")
    logger.info("  3. torch.view_as_complex -> real tensor operations")
    logger.info("  4. torch.jit.script -> removed (direct forward)")
    logger.info("  5. Complex arithmetic -> real tensor arithmetic")
    logger.info("  6. tensor.unfold -> explicit slicing")
    logger.info("  7-12. Additional einsum patterns in multiframe")
    logger.info("=" * 60)
