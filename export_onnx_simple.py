#!/usr/bin/env python3
"""
Simple ONNX export script for DeepFilterNet
"""

import os
import sys
import torch
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "DeepFilterNet" / "df"))

from loguru import logger
from enhance import init_df
from model import ModelParams
from model_converter_onnx import convert_model_to_onnx_compatible

def export_onnx(model_dir: str, output_dir: str, opset_version: int = 15):
    """Export DeepFilterNet model to ONNX format"""

    logger.info("=" * 70)
    logger.info("ONNX EXPORT - DeepFilterNet")
    logger.info("=" * 70)

    # Load model
    logger.info(f"\nLoading model from: {model_dir}")
    model, df_state, suffix, epoch = init_df(
        model_dir,
        config_allow_defaults=True,
        epoch='best',
        log_file=None,
    )

    model.eval()
    model = model.to("cpu")

    p = ModelParams()

    # Convert to ONNX-compatible version
    logger.info("\nConverting model to ONNX-compatible version...")
    model_onnx = convert_model_to_onnx_compatible(model, inplace=False)
    model_onnx.eval()

    # Create export directory
    os.makedirs(output_dir, exist_ok=True)

    # Export each component
    logger.info("\n" + "=" * 70)
    logger.info("EXPORTING MODELS")
    logger.info("=" * 70)

    # 1. Encoder
    logger.info("\n[1/3] Exporting Encoder...")
    B, T, C, F = 1, 100, 1, p.nb_erb
    feat_erb = torch.randn(B, C, T, F)
    # feat_spec has 2 channels (real/imag): [B, 2, T, F]
    feat_spec = torch.randn(B, 2, T, p.nb_df)

    enc_path = os.path.join(output_dir, "enc.onnx")
    logger.info(f"  Input shapes: feat_erb={tuple(feat_erb.shape)}, feat_spec={tuple(feat_spec.shape)}")

    with torch.no_grad():
        enc_out = model_onnx.enc(feat_erb, feat_spec)

    # Handle tuple output
    if isinstance(enc_out, tuple):
        enc_out = enc_out[0]  # Get embeddings (first output)

    logger.info(f"  Output shape: {tuple(enc_out.shape)}")
    logger.info(f"  Exporting to: {enc_path}")

    torch.onnx.export(
        model=model_onnx.enc,
        f=enc_path,
        args=(feat_erb, feat_spec),
        input_names=["feat_erb", "feat_spec"],
        output_names=["enc_out"],
        dynamic_axes={
            "feat_erb": {0: "batch"},
            "feat_spec": {0: "batch"},
            "enc_out": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,  # Use legacy export (TorchScript-based)
    )
    logger.info("  ✓ Encoder exported")

    # 2. ERB Decoder
    logger.info("\n[2/3] Exporting ERB Decoder...")
    # enc_out shape is [B, C, T, E] - get E from last dimension
    E = enc_out.shape[-1]
    emb = torch.randn(B, T, E)

    erb_dec_path = os.path.join(output_dir, "erb_dec.onnx")
    logger.info(f"  Input shape: emb={tuple(emb.shape)}")

    with torch.no_grad():
        erb_out = model_onnx.erb_dec(emb)

    logger.info(f"  Output shape: {tuple(erb_out.shape)}")
    logger.info(f"  Exporting to: {erb_dec_path}")

    torch.onnx.export(
        model=model_onnx.erb_dec,
        f=erb_dec_path,
        args=(emb,),
        input_names=["emb"],
        output_names=["erb_out"],
        dynamic_axes={
            "emb": {0: "batch"},
            "erb_out": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )
    logger.info("  ✓ ERB Decoder exported")

    # 3. DF Decoder
    logger.info("\n[3/3] Exporting DF Decoder...")
    df_dec_path = os.path.join(output_dir, "df_dec.onnx")
    logger.info(f"  Input shape: emb={tuple(emb.shape)}")

    with torch.no_grad():
        df_alpha, df_coefs = model_onnx.df_dec(emb, erb_out)

    logger.info(f"  Output shapes: alpha={tuple(df_alpha.shape)}, coefs={tuple(df_coefs.shape)}")
    logger.info(f"  Exporting to: {df_dec_path}")

    torch.onnx.export(
        model=model_onnx.df_dec,
        f=df_dec_path,
        args=(emb, erb_out),
        input_names=["emb", "erb_out"],
        output_names=["alpha", "coefs"],
        dynamic_axes={
            "emb": {0: "batch"},
            "erb_out": {0: "batch"},
            "alpha": {0: "batch"},
            "coefs": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )
    logger.info("  ✓ DF Decoder exported")

    # Validation
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION")
    logger.info("=" * 70)

    import onnx
    import onnxruntime as ort

    for name, path in [("Encoder", enc_path), ("ERB Decoder", erb_dec_path), ("DF Decoder", df_dec_path)]:
        logger.info(f"\nValidating {name}...")

        # Check ONNX model
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        logger.info(f"  ✓ ONNX check passed")

        # Create inference session
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        logger.info(f"  ✓ ONNX Runtime can load model")
        logger.info(f"  File size: {os.path.getsize(path) / 1024:.1f} KB")

    logger.info("\n" + "=" * 70)
    logger.info("EXPORT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nModels exported to: {output_dir}")
    logger.info(f"  - enc.onnx (Encoder)")
    logger.info(f"  - erb_dec.onnx (ERB Decoder)")
    logger.info(f"  - df_dec.onnx (DF Decoder)")
    logger.info(f"\nOpset version: {opset_version}")
    logger.info(f"All models validated successfully!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export DeepFilterNet to ONNX")
    parser.add_argument("model_dir", help="Path to model directory")
    parser.add_argument("output_dir", help="Output directory for ONNX models")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version (use 11 or 13 for QNN)")

    args = parser.parse_args()

    export_onnx(args.model_dir, args.output_dir, args.opset)
