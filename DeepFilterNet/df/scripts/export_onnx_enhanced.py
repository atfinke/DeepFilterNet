"""
Enhanced ONNX Export Script for DeepFilterNet

This script exports DeepFilterNet models to ONNX format with the following improvements:
1. Automatic conversion of incompatible operations to ONNX-friendly alternatives
2. Comprehensive validation against original PyTorch model
3. Support for multiple opset versions
4. Detailed logging and error reporting
5. Export metadata for QNN conversion

Usage:
    python -m df.scripts.export_onnx_enhanced <model_dir> <export_dir> [options]

Example:
    python -m df.scripts.export_onnx_enhanced ./pretrained_models/DeepFilterNet2 ./onnx_models \\
        --opset 15 --validate --simplify
"""

import argparse
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnx.checker
import onnx.helper
import onnxruntime as ort
import torch
from loguru import logger
from torch import Tensor

from df.enhance import ModelParams, df_features, get_model_basedir, init_df
from df.io import get_test_sample
from df.model_converter_onnx import (
    convert_model_to_onnx_compatible,
    count_replaced_operations,
    print_conversion_summary,
    validate_conversion,
)
from libdf import DF


def export_onnx_model(
    path: str,
    model: torch.nn.Module,
    inputs: Tuple[Tensor, ...],
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Dict[str, Dict[int, str]],
    opset_version: int = 15,
    check: bool = True,
    simplify: bool = True,
) -> Tuple[Tensor, ...]:
    """
    Export a PyTorch model to ONNX format.

    Args:
        path: Output ONNX file path
        model: PyTorch model to export
        inputs: Tuple of input tensors
        input_names: List of input names
        output_names: List of output names
        dynamic_axes: Dictionary specifying dynamic axes
        opset_version: ONNX opset version (15+ recommended)
        check: Whether to validate ONNX model
        simplify: Whether to simplify ONNX graph

    Returns:
        Tuple of output tensors from the model
    """
    export_dir = os.path.dirname(path)
    if not os.path.isdir(export_dir):
        logger.info(f"Creating export directory: {export_dir}")
        os.makedirs(export_dir, exist_ok=True)

    model_name = os.path.splitext(os.path.basename(path))[0]
    logger.info(f"\nExporting model '{model_name}' to {export_dir}")
    logger.info(f"  Input shapes: {[(name, tuple(t.shape)) for name, t in zip(input_names, inputs)]}")

    # Get outputs for validation
    with torch.no_grad():
        outputs = model(*inputs)

    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    logger.info(f"  Output shapes: {[(name, tuple(t.shape)) for name, t in zip(output_names, outputs)]}")
    logger.info(f"  Dynamic axes: {dynamic_axes}")
    logger.info(f"  Opset version: {opset_version}")

    # Export to ONNX (no TorchScript compilation - direct export)
    logger.info("  Exporting to ONNX...")
    torch.onnx.export(
        model=model,
        f=path,
        args=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        export_params=True,
    )
    logger.info(f"  ✓ Exported to {path}")

    # Validate ONNX model
    if check:
        logger.info("  Validating ONNX model...")
        try:
            onnx_model = onnx.load(path)
            onnx.checker.check_model(onnx_model, full_check=True)
            logger.info("  ✓ ONNX model is valid")

            # Check output accuracy
            input_dict = {k: v for k, v in zip(input_names, inputs)}
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            onnx_outputs = sess.run(output_names, {k: v.numpy() for k, v in input_dict.items()})

            max_error = 0.0
            for name, pytorch_out, onnx_out in zip(output_names, outputs, onnx_outputs):
                error = np.max(np.abs(pytorch_out.numpy() - onnx_out))
                max_error = max(max_error, error)
                logger.info(f"  Output '{name}': max error = {error:.2e}")

            if max_error > 1e-3:
                logger.warning(f"  ⚠ Large output error detected: {max_error:.2e}")
            else:
                logger.info(f"  ✓ Output accuracy validated (max error: {max_error:.2e})")

        except Exception as e:
            logger.error(f"  ✗ ONNX validation failed: {e}")
            raise

    # Simplify ONNX graph
    if simplify:
        logger.info("  Simplifying ONNX graph...")
        try:
            import onnxsim

            onnx_model = onnx.load(path)
            input_shapes = {k: tuple(v.shape) for k, v in zip(input_names, inputs)}
            input_data = {k: v.numpy() for k, v in zip(input_names, inputs)}

            model_simp, check_ok = onnxsim.simplify(
                onnx_model,
                input_data=input_data,
                test_input_shapes=input_shapes,
            )

            if check_ok:
                onnx.save(model_simp, path)
                logger.info(f"  ✓ Simplified and saved to {path}")
            else:
                logger.warning("  ⚠ Simplification check failed, keeping original")

        except Exception as e:
            logger.warning(f"  ⚠ Simplification failed: {e}")

    return outputs


def export_deepfilternet_onnx(
    model,
    export_dir: str,
    df_state: DF,
    opset: int = 15,
    check: bool = True,
    simplify: bool = True,
) -> None:
    """
    Export DeepFilterNet model components to ONNX format.

    Exports three separate models:
    1. enc.onnx - Encoder
    2. erb_dec.onnx - ERB Decoder
    3. df_dec.onnx - DF Decoder

    Args:
        model: DeepFilterNet model
        export_dir: Directory to save ONNX models
        df_state: DF state object
        opset: ONNX opset version
        check: Whether to validate exports
        simplify: Whether to simplify ONNX graphs
    """
    model = model.cpu()
    model.eval()

    p = ModelParams()
    audio = torch.randn((1, 1 * p.sr))
    spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device="cpu")

    logger.info("\n" + "=" * 70)
    logger.info("EXPORTING DEEPFILTERNET TO ONNX")
    logger.info("=" * 70)

    # Export encoder
    logger.info("\n[1/3] ENCODER")
    logger.info("-" * 70)
    feat_spec_transposed = feat_spec.transpose(1, 4).squeeze(4)  # Move re/im to channel axis
    path = os.path.join(export_dir, "enc.onnx")
    inputs = (feat_erb, feat_spec_transposed)
    input_names = ["feat_erb", "feat_spec"]
    dynamic_axes = {
        "feat_erb": {2: "T"},
        "feat_spec": {2: "T"},
        "e0": {2: "T"},
        "e1": {2: "T"},
        "e2": {2: "T"},
        "e3": {2: "T"},
        "emb": {1: "T"},
        "c0": {2: "T"},
        "lsnr": {1: "T"},
    }
    output_names = ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"]

    e0, e1, e2, e3, emb, c0, lsnr = export_onnx_model(
        path,
        model.enc,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        check=check,
        simplify=simplify,
    )

    # Save test data
    np.savez_compressed(
        os.path.join(export_dir, "enc_input.npz"),
        feat_erb=feat_erb.numpy(),
        feat_spec=feat_spec_transposed.numpy(),
    )
    np.savez_compressed(
        os.path.join(export_dir, "enc_output.npz"),
        e0=e0.numpy(),
        e1=e1.numpy(),
        e2=e2.numpy(),
        e3=e3.numpy(),
        emb=emb.numpy(),
        c0=c0.numpy(),
        lsnr=lsnr.numpy(),
    )

    # Export ERB decoder
    logger.info("\n[2/3] ERB DECODER")
    logger.info("-" * 70)
    inputs = (emb.clone(), e3, e2, e1, e0)
    input_names = ["emb", "e3", "e2", "e1", "e0"]
    output_names = ["m"]
    dynamic_axes = {
        "emb": {1: "T"},
        "e3": {2: "T"},
        "e2": {2: "T"},
        "e1": {2: "T"},
        "e0": {2: "T"},
        "m": {2: "T"},
    }
    path = os.path.join(export_dir, "erb_dec.onnx")

    (m,) = export_onnx_model(
        path,
        model.erb_dec,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        check=check,
        simplify=simplify,
    )

    np.savez_compressed(
        os.path.join(export_dir, "erb_dec_input.npz"),
        emb=emb.numpy(),
        e0=e0.numpy(),
        e1=e1.numpy(),
        e2=e2.numpy(),
        e3=e3.numpy(),
    )
    np.savez_compressed(os.path.join(export_dir, "erb_dec_output.npz"), m=m.numpy())

    # Export DF decoder
    logger.info("\n[3/3] DF DECODER")
    logger.info("-" * 70)
    inputs = (emb.clone(), c0)
    input_names = ["emb", "c0"]
    output_names = ["coefs"]
    dynamic_axes = {
        "emb": {1: "T"},
        "c0": {2: "T"},
        "coefs": {1: "T"},
    }
    path = os.path.join(export_dir, "df_dec.onnx")

    (coefs,) = export_onnx_model(
        path,
        model.df_dec,
        inputs=inputs,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        check=check,
        simplify=simplify,
    )

    np.savez_compressed(
        os.path.join(export_dir, "df_dec_input.npz"), emb=emb.numpy(), c0=c0.numpy()
    )
    np.savez_compressed(os.path.join(export_dir, "df_dec_output.npz"), coefs=coefs.numpy())

    logger.info("\n" + "=" * 70)
    logger.info("ONNX EXPORT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nExported models:")
    logger.info(f"  • {os.path.join(export_dir, 'enc.onnx')}")
    logger.info(f"  • {os.path.join(export_dir, 'erb_dec.onnx')}")
    logger.info(f"  • {os.path.join(export_dir, 'df_dec.onnx')}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced ONNX export for DeepFilterNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_base_dir", help="Path to model directory")
    parser.add_argument("export_dir", help="Directory for exporting ONNX models")
    parser.add_argument("--opset", type=int, default=15, help="ONNX opset version (default: 15)")
    parser.add_argument(
        "--no-check", action="store_false", dest="check", help="Don't validate ONNX models"
    )
    parser.add_argument(
        "--simplify", action="store_true", help="Simplify ONNX models using onnxsim"
    )
    parser.add_argument("--pf", action="store_true", help="Enable post-filter")
    parser.add_argument("--epoch", type=str, default="best", help="Model epoch to load")
    parser.add_argument(
        "--no-convert",
        action="store_true",
        help="Skip automatic conversion to ONNX-compatible operations",
    )
    parser.add_argument(
        "--validate-conversion",
        action="store_true",
        help="Validate converted model against original",
    )

    args = parser.parse_args()

    # Initialize model
    logger.info("Loading model...")
    model, df_state, _, epoch = init_df(
        args.model_base_dir,
        post_filter=args.pf,
        config_allow_defaults=True,
        epoch=args.epoch,
    )

    # Convert model to ONNX-compatible version
    if not args.no_convert:
        logger.info("\n" + "=" * 70)
        logger.info("CONVERTING MODEL TO ONNX-COMPATIBLE VERSION")
        logger.info("=" * 70)

        # Validate conversion if requested
        if args.validate_conversion:
            logger.info("\nGenerating test inputs for validation...")
            p = ModelParams()
            audio = torch.randn((1, 1 * p.sr))
            spec, feat_erb, feat_spec = df_features(audio, df_state, p.nb_df, device="cpu")
            feat_spec_transposed = feat_spec.transpose(1, 4).squeeze(4)

            test_inputs_enc = {"feat_erb": feat_erb, "feat_spec": feat_spec_transposed}

            logger.info("\nValidating encoder conversion...")
            original_enc = model.enc
            converted_enc = convert_model_to_onnx_compatible(original_enc, inplace=False)
            validate_conversion(original_enc, converted_enc, (feat_erb, feat_spec_transposed))

        # Convert full model
        model = convert_model_to_onnx_compatible(model, inplace=True)

        # Count and report conversions
        counts = count_replaced_operations(model)
        print_conversion_summary(counts)

    # Create export directory
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export models
    export_deepfilternet_onnx(
        model,
        export_dir,
        df_state=df_state,
        opset=args.opset,
        check=args.check,
        simplify=args.simplify,
    )

    # Copy config file
    model_base_dir = get_model_basedir(args.model_base_dir)
    if model_base_dir != args.export_dir:
        config_src = os.path.join(model_base_dir, "config.ini")
        if os.path.exists(config_src):
            import shutil
            shutil.copyfile(config_src, os.path.join(args.export_dir, "config.ini"))
            logger.info(f"\nCopied config.ini to {args.export_dir}")

    # Write version file
    model_name = Path(model_base_dir).name
    version_file = os.path.join(args.export_dir, "version.txt")
    with open(version_file, "w") as f:
        f.write(f"{model_name}_epoch_{epoch}_onnx_enhanced")
    logger.info(f"Wrote version info to {version_file}")

    # Create tarball
    tar_name = export_dir / (Path(model_base_dir).name + "_onnx_enhanced.tar.gz")
    logger.info(f"\nCreating tarball: {tar_name}")
    with tarfile.open(tar_name, mode="w:gz") as f:
        f.add(os.path.join(args.export_dir, "enc.onnx"), arcname="enc.onnx")
        f.add(os.path.join(args.export_dir, "erb_dec.onnx"), arcname="erb_dec.onnx")
        f.add(os.path.join(args.export_dir, "df_dec.onnx"), arcname="df_dec.onnx")
        f.add(os.path.join(args.export_dir, "config.ini"), arcname="config.ini")
        f.add(os.path.join(args.export_dir, "version.txt"), arcname="version.txt")
    logger.info(f"✓ Created {tar_name}")

    logger.info("\n" + "=" * 70)
    logger.info("ALL DONE!")
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("1. Test ONNX models with ONNX Runtime")
    logger.info("2. Convert ONNX models to QNN format using qnn-onnx-converter")
    logger.info("3. Quantize models using qnn-quantizer")
    logger.info("4. Deploy to Hexagon NPU\n")


if __name__ == "__main__":
    main()
