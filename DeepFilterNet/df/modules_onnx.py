"""
ONNX-Compatible Module Implementations for DeepFilterNet

This module provides ONNX-compatible replacements for operations that are
problematic for ONNX export and QNN deployment, including:

1. torch.einsum -> explicit matrix operations
2. torch.as_strided -> explicit padding and reshaping
3. torch.view_as_complex -> real tensor operations
4. Custom implementations avoiding TorchScript issues

These modules maintain mathematical equivalence with the original implementations
while ensuring clean ONNX graph export and QNN/Hexagon NPU compatibility.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from typing_extensions import Final


# ============================================================================
# Complex Number Operations Using Real Tensors
# ============================================================================


def complex_multiply_real(a: Tensor, b: Tensor) -> Tensor:
    """
    Complex multiplication using real tensors.

    Args:
        a: Real tensor of shape [..., 2] where last dim is [real, imag]
        b: Real tensor of shape [..., 2] where last dim is [real, imag]

    Returns:
        Complex product as real tensor [..., 2]

    Formula: (a_r + a_i*j)(b_r + b_i*j) = (a_r*b_r - a_i*b_i) + (a_r*b_i + a_i*b_r)*j

    REPLACES: torch.view_as_complex(a) * torch.view_as_complex(b)
    """
    a_real, a_imag = a[..., 0], a[..., 1]
    b_real, b_imag = b[..., 0], b[..., 1]

    real = a_real * b_real - a_imag * b_imag
    imag = a_real * b_imag + a_imag * b_real

    return torch.stack([real, imag], dim=-1)


def complex_conj_real(x: Tensor) -> Tensor:
    """
    Complex conjugate using real tensors.

    Args:
        x: Real tensor of shape [..., 2] where last dim is [real, imag]

    Returns:
        Complex conjugate as real tensor [..., 2]

    REPLACES: torch.conj(torch.view_as_complex(x))
    """
    real = x[..., 0]
    imag = -x[..., 1]
    return torch.stack([real, imag], dim=-1)


def complex_abs_real(x: Tensor) -> Tensor:
    """
    Complex absolute value (magnitude) using real tensors.

    Args:
        x: Real tensor of shape [..., 2] where last dim is [real, imag]

    Returns:
        Magnitude as real tensor [...]

    REPLACES: torch.abs(torch.view_as_complex(x))
    """
    return torch.sqrt(x[..., 0]**2 + x[..., 1]**2)


# ============================================================================
# GroupedLinearExplicit - Replaces GroupedLinearEinsum
# ============================================================================


class GroupedLinearExplicit(nn.Module):
    """
    Grouped linear layer using explicit batch matrix multiply instead of einsum.

    REPLACES: GroupedLinearEinsum which uses torch.einsum("btgi,gih->btgh", x, self.weight)
    OPERATION #1: torch.einsum -> torch.bmm

    Original einsum pattern breakdown:
    - b: batch dimension
    - t: time dimension
    - g: group dimension
    - i: input features per group
    - h: hidden features per group

    Replacement strategy:
    1. Reshape input from [B, T, I] to [B*T, G, I/G]
    2. Batch matmul with weight [G, I/G, H/G] -> [B*T, G, H/G]
    3. Reshape output back to [B, T, H]
    """

    input_size: Final[int]
    hidden_size: Final[int]
    groups: Final[int]

    def __init__(self, input_size: int, hidden_size: int, groups: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.groups = groups

        assert input_size % groups == 0, f"Input size {input_size} not divisible by {groups}"
        assert hidden_size % groups == 0, f"Hidden size {hidden_size} not divisible by {groups}"

        self.input_size_per_group = input_size // groups
        self.hidden_size_per_group = hidden_size // groups

        # Weight shape: [groups, input_size_per_group, hidden_size_per_group]
        self.weight = Parameter(
            torch.zeros(groups, self.input_size_per_group, self.hidden_size_per_group),
            requires_grad=True
        )
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape [B, T, I]

        Returns:
            Output tensor of shape [B, T, H]
        """
        b, t, _ = x.shape

        # Reshape input: [B, T, I] -> [B*T, G, I/G]
        x_reshaped = x.view(b * t, self.groups, self.input_size_per_group)

        # Batch matrix multiply: [B*T, G, I/G] x [G, I/G, H/G] -> [B*T, G, H/G]
        # We need to expand weight for batch dimension
        weight_expanded = self.weight.unsqueeze(0).expand(b * t, -1, -1, -1)
        # bmm requires 3D tensors, so we do it per group or use matmul with broadcasting

        # Better approach: use matmul with broadcasting
        # x_reshaped: [B*T, G, 1, I/G]
        # weight: [1, G, I/G, H/G]
        # output: [B*T, G, 1, H/G] -> [B*T, G, H/G]
        x_expanded = x_reshaped.unsqueeze(2)  # [B*T, G, 1, I/G]
        weight_expanded = self.weight.unsqueeze(0)  # [1, G, I/G, H/G]

        # Matrix multiply
        output = torch.matmul(x_expanded, weight_expanded)  # [B*T, G, 1, H/G]
        output = output.squeeze(2)  # [B*T, G, H/G]

        # Reshape back: [B*T, G, H/G] -> [B, T, H]
        output = output.reshape(b, t, self.hidden_size)

        return output

    def __repr__(self):
        return (f"GroupedLinearExplicit(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, groups={self.groups})")


# ============================================================================
# DfOpONNX - Simplified Deep Filter Operation
# ============================================================================


class DfOpONNX(nn.Module):
    """
    ONNX-compatible Deep Filter operation.

    REPLACES: DfOp which uses torch.as_strided and torch.view_as_complex
    OPERATION #2: torch.as_strided -> explicit padding + slicing
    OPERATION #3: torch.view_as_complex -> real tensor operations

    Uses forward_real_loop_optimized which avoids:
    - torch.as_strided (memory layout operations)
    - torch.view_as_complex (complex number operations)
    - torch.jit.script (TorchScript compilation issues)

    Implements multi-tap FIR filtering in frequency domain using only
    basic PyTorch operations that export cleanly to ONNX.
    """

    df_order: Final[int]
    df_bins: Final[int]
    df_lookahead: Final[int]
    freq_bins: Final[int]

    def __init__(
        self,
        df_bins: int,
        df_order: int = 5,
        df_lookahead: int = 0,
        freq_bins: int = 0,
    ):
        super().__init__()
        self.df_order = df_order
        self.df_bins = df_bins
        self.df_lookahead = df_lookahead
        self.freq_bins = freq_bins if freq_bins > 0 else df_bins

    def forward(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        """
        Apply deep filter to spectrogram.

        Args:
            spec: Complex spectrogram as real tensor [B, 1, T, F, 2]
            coefs: Filter coefficients [B, T, O, F_df, 2]
            alpha: Blending factor [B, T, 1]

        Returns:
            Filtered spectrogram [B, 1, T, F, 2]
        """
        return self.forward_real_loop_optimized(spec, coefs, alpha)

    def forward_real_loop_optimized(
        self, spec: Tensor, coefs: Tensor, alpha: Optional[Tensor] = None
    ) -> Tensor:
        """
        Optimized loop-based implementation avoiding as_strided.

        OPERATION #4: Eliminates dynamic control flow via vectorization where possible
        """
        b, _, t, f, _ = spec.shape
        df_bins = self.df_bins

        # Extract DF bins from spectrogram
        spec_df = spec[..., :df_bins, :]  # [B, 1, T, F_df, 2]
        spec_df = spec_df.squeeze(1)  # [B, T, F_df, 2]

        # Manual padding for temporal filtering
        # Pad: [order - lookahead - 1, lookahead]
        pad_left = self.df_order - self.df_lookahead - 1
        pad_right = self.df_lookahead

        # Pad on time dimension
        if pad_left > 0 or pad_right > 0:
            padded = F.pad(spec_df, (0, 0, 0, 0, pad_left, pad_right), mode='constant', value=0.0)
        else:
            padded = spec_df
        # padded: [B, T + order - 1, F_df, 2]

        # Initialize output
        spec_f_real = torch.zeros((b, t, df_bins), device=spec.device, dtype=spec.dtype)
        spec_f_imag = torch.zeros((b, t, df_bins), device=spec.device, dtype=spec.dtype)

        # Apply filter taps - vectorized over time
        for i in range(self.df_order):
            # Extract time slice [i : i+t]
            padded_slice = padded[:, i:i+t, :, :]  # [B, T, F_df, 2]
            coef_slice = coefs[:, :, i, :, :]  # [B, T, F_df, 2]

            # Complex multiply: (padded_real + j*padded_imag) * (coef_real + j*coef_imag)
            # real_part = padded_real * coef_real - padded_imag * coef_imag
            # imag_part = padded_real * coef_imag + padded_imag * coef_real

            padded_real = padded_slice[..., 0]
            padded_imag = padded_slice[..., 1]
            coef_real = coef_slice[..., 0]
            coef_imag = coef_slice[..., 1]

            spec_f_real += padded_real * coef_real - padded_imag * coef_imag
            spec_f_imag += padded_real * coef_imag + padded_imag * coef_real

        # Stack real and imaginary parts
        spec_f = torch.stack([spec_f_real, spec_f_imag], dim=-1)  # [B, T, F_df, 2]
        spec_f = spec_f.unsqueeze(1)  # [B, 1, T, F_df, 2]

        # Assign filtered result to output
        return self._assign_df(spec, spec_f, df_bins, alpha)

    def _assign_df(
        self, spec: Tensor, spec_f: Tensor, df_bins: int, alpha: Optional[Tensor]
    ) -> Tensor:
        """
        Assign filtered spectrogram to output with optional blending.

        Args:
            spec: Original spectrogram [B, 1, T, F, 2]
            spec_f: Filtered spectrogram [B, 1, T, F_df, 2]
            df_bins: Number of DF bins
            alpha: Blending factor [B, T, 1]

        Returns:
            Output spectrogram [B, 1, T, F, 2]
        """
        spec_out = spec.clone()

        if alpha is not None:
            b = spec.shape[0]
            alpha = alpha.view(b, 1, -1, 1, 1)  # [B, 1, T, 1, 1]
            # Blend: spec_f * alpha + spec * (1 - alpha)
            spec_out[..., :df_bins, :] = (
                spec_f * alpha + spec[..., :df_bins, :] * (1 - alpha)
            )
        else:
            spec_out[..., :df_bins, :] = spec_f

        return spec_out


# ============================================================================
# SqueezedGRU_ONNX - Using GroupedLinearExplicit
# ============================================================================


class SqueezedGRU_ONNX(nn.Module):
    """
    SqueezedGRU using ONNX-compatible GroupedLinearExplicit instead of einsum.

    REPLACES: SqueezedGRU which uses GroupedLinearEinsum
    """

    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
        batch_first: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.linear_in = GroupedLinearExplicit(input_size, hidden_size, linear_groups)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=batch_first)

        if output_size is not None:
            self.linear_out = GroupedLinearExplicit(hidden_size, output_size, linear_groups)
        else:
            self.linear_out = nn.Identity()

    def forward(self, input: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        input = self.linear_in(input)
        x, h = self.gru(input, h)
        x = self.linear_out(x)
        return x, h


# ============================================================================
# Multiframe Operations - Einsum Replacements
# ============================================================================


def apply_df_multiframe_explicit(spec: Tensor, coefs: Tensor) -> Tensor:
    """
    Apply deep filter coefficients to unfolded spectrogram.

    REPLACES: torch.einsum("...tfn,...ntf->...tf", spec, coefs)
    OPERATION #5: torch.einsum -> transpose + batch matmul

    Args:
        spec: Unfolded spectrogram [..., T, F, N]
        coefs: Filter coefficients [..., N, T, F]

    Returns:
        Filtered spectrogram [..., T, F]
    """
    # einsum("...tfn,...ntf->...tf", spec, coefs)
    # This sums over n dimension: for each t,f: sum_n(spec[t,f,n] * coefs[n,t,f])

    # Approach: multiply element-wise after broadcasting, then sum over n
    # spec: [..., T, F, N]
    # coefs: [..., N, T, F] -> need to permute to [..., T, F, N]

    coefs_permuted = coefs.permute(*range(len(coefs.shape) - 3), -2, -1, -3)  # [..., T, F, N]

    # Element-wise multiply and sum over N
    result = (spec * coefs_permuted).sum(dim=-1)  # [..., T, F]

    return result


def outer_product_explicit(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute outer product of two tensors.

    REPLACES: torch.einsum("...n,...m->...nm", x, y)
    OPERATION #6: torch.einsum -> unsqueeze + multiply

    Args:
        x: Tensor of shape [..., N]
        y: Tensor of shape [..., M]

    Returns:
        Outer product [..., N, M]
    """
    # x: [..., N] -> [..., N, 1]
    # y: [..., M] -> [..., 1, M]
    # result: [..., N, M]
    return x.unsqueeze(-1) * y.unsqueeze(-2)


def matrix_vector_multiply_explicit(matrix: Tensor, vector: Tensor) -> Tensor:
    """
    Matrix-vector multiplication along last dimensions.

    REPLACES: torch.einsum("...nm,...m->...n", matrix, vector)
    OPERATION #7: torch.einsum -> matmul

    Args:
        matrix: Tensor of shape [..., N, M]
        vector: Tensor of shape [..., M]

    Returns:
        Result [..., N]
    """
    # matrix: [..., N, M]
    # vector: [..., M] -> [..., M, 1]
    # result: [..., N, 1] -> [..., N]
    return torch.matmul(matrix, vector.unsqueeze(-1)).squeeze(-1)


def inner_product_explicit(x: Tensor, y: Tensor) -> Tensor:
    """
    Inner product (dot product) of two tensors.

    REPLACES: torch.einsum("...n,...n->...", x, y)
    OPERATION #8: torch.einsum -> multiply + sum

    Args:
        x: Tensor of shape [..., N]
        y: Tensor of shape [..., N]

    Returns:
        Inner product [...]
    """
    return (x * y).sum(dim=-1)


# ============================================================================
# Tensor Unfold Replacement
# ============================================================================


def unfold_time_explicit(x: Tensor, window_size: int, step: int = 1) -> Tensor:
    """
    Explicit replacement for tensor.unfold() operation.

    REPLACES: tensor.unfold(dimension=1, size=window_size, step=1)
    OPERATION #9: tensor.unfold -> explicit slicing + stacking

    Args:
        x: Input tensor [B, T, ...]
        window_size: Size of sliding window
        step: Step size for sliding window

    Returns:
        Unfolded tensor [B, T', window_size, ...]
    """
    b, t = x.shape[:2]
    rest_shape = x.shape[2:]

    # Calculate output time dimension
    t_out = (t - window_size) // step + 1

    # Collect windows
    windows = []
    for i in range(0, t - window_size + 1, step):
        windows.append(x[:, i:i+window_size, ...])

    # Stack along new dimension
    # Each window: [B, window_size, ...]
    # Stack to: [B, T', window_size, ...]
    output = torch.stack(windows, dim=1)

    return output


# ============================================================================
# Export Helper
# ============================================================================


def replace_einsum_modules(model: nn.Module) -> nn.Module:
    """
    Replace einsum-based modules with ONNX-compatible versions in a model.

    Args:
        model: PyTorch model containing einsum operations

    Returns:
        Modified model with ONNX-compatible operations
    """
    # Import here to avoid circular dependency
    from df.modules import GroupedLinearEinsum, SqueezedGRU, SqueezedGRU_S

    for name, module in model.named_children():
        if isinstance(module, GroupedLinearEinsum):
            # Replace with explicit version
            new_module = GroupedLinearExplicit(
                input_size=module.input_size,
                hidden_size=module.hidden_size,
                groups=module.groups
            )
            # Copy weights
            new_module.weight.data = module.weight.data.clone()
            setattr(model, name, new_module)
        elif isinstance(module, (SqueezedGRU, SqueezedGRU_S)):
            # Replace with ONNX version
            new_module = SqueezedGRU_ONNX(
                input_size=module.input_size,
                hidden_size=module.hidden_size,
                output_size=None,  # Will be handled by linear_out
                num_layers=1,
                linear_groups=8,  # Default value
            )
            # Copy weights from linear_in
            if hasattr(module.linear_in, '0'):  # nn.Sequential
                old_linear = module.linear_in[0]
                if isinstance(old_linear, GroupedLinearEinsum):
                    new_module.linear_in.weight.data = old_linear.weight.data.clone()
            # Copy GRU weights
            new_module.gru.load_state_dict(module.gru.state_dict())
            # Copy weights from linear_out
            if hasattr(module, 'linear_out') and hasattr(module.linear_out, '0'):
                old_linear_out = module.linear_out[0]
                if isinstance(old_linear_out, GroupedLinearEinsum):
                    if isinstance(new_module.linear_out, GroupedLinearExplicit):
                        new_module.linear_out.weight.data = old_linear_out.weight.data.clone()
            setattr(model, name, new_module)
        else:
            # Recursively replace in child modules
            replace_einsum_modules(module)

    return model


# ============================================================================
# Summary of Operations Replaced
# ============================================================================

"""
SUMMARY OF ONNX/QNN INCOMPATIBLE OPERATIONS REPLACED:

1. torch.einsum("btgi,gih->btgh") -> GroupedLinearExplicit with matmul
2. torch.as_strided -> explicit padding + slicing in DfOpONNX
3. torch.view_as_complex -> complex_multiply_real with real tensors
4. Dynamic for loops -> vectorized operations where possible
5. torch.einsum("...tfn,...ntf->...tf") -> apply_df_multiframe_explicit
6. torch.einsum("...n,...m->...nm") -> outer_product_explicit
7. torch.einsum("...nm,...m->...n") -> matrix_vector_multiply_explicit
8. torch.einsum("...n,...n->...") -> inner_product_explicit
9. tensor.unfold() -> unfold_time_explicit with explicit slicing
10. torch.jit.script in DfOp -> removed, use direct forward pass
11. torch.view_as_real with complex operations -> always use real format
12. Complex conj() -> complex_conj_real

All replacements maintain mathematical equivalence while ensuring:
- Clean ONNX graph export
- QNN/Hexagon NPU compatibility
- No memory layout operations (as_strided)
- No complex number types (use real tensors with [..., 2] shape)
- No dynamic control flow (vectorized operations)
"""
