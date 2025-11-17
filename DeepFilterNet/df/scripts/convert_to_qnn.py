"""
QNN Conversion Script for DeepFilterNet

This script converts ONNX models to QNN format for deployment on Qualcomm Hexagon NPU.

Prerequisites:
    - Qualcomm QNN SDK installed and configured
    - ONNX models exported using export_onnx_enhanced.py
    - Environment variables: QNN_SDK_ROOT

Usage:
    python -m df.scripts.convert_to_qnn <onnx_dir> <qnn_output_dir> [options]

Example:
    export QNN_SDK_ROOT=/path/to/qnn/sdk
    python -m df.scripts.convert_to_qnn ./onnx_models ./qnn_models --quantize --backend hexagon
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger


class QNNConverter:
    """
    Converter for ONNX models to QNN format.
    """

    def __init__(self, qnn_sdk_root: Optional[str] = None):
        """
        Initialize QNN converter.

        Args:
            qnn_sdk_root: Path to QNN SDK root directory
        """
        self.qnn_sdk_root = qnn_sdk_root or os.environ.get("QNN_SDK_ROOT")

        if not self.qnn_sdk_root:
            raise RuntimeError(
                "QNN_SDK_ROOT not found. Please set it via:\n"
                "  export QNN_SDK_ROOT=/path/to/qnn/sdk\n"
                "or pass --qnn-sdk-root argument"
            )

        self.qnn_sdk_root = Path(self.qnn_sdk_root)
        if not self.qnn_sdk_root.exists():
            raise FileNotFoundError(f"QNN SDK not found at: {self.qnn_sdk_root}")

        # Find QNN tools
        self.qnn_converter = self._find_tool("qnn-onnx-converter")
        self.qnn_net_run = self._find_tool("qnn-net-run")
        self.qnn_context_binary_generator = self._find_tool("qnn-context-binary-generator")

        logger.info(f"QNN SDK found at: {self.qnn_sdk_root}")

    def _find_tool(self, tool_name: str) -> Path:
        """Find QNN tool in SDK."""
        possible_paths = [
            self.qnn_sdk_root / "bin" / "x86_64-linux-clang" / tool_name,
            self.qnn_sdk_root / "bin" / tool_name,
        ]

        for path in possible_paths:
            if path.exists():
                logger.debug(f"Found {tool_name} at: {path}")
                return path

        raise FileNotFoundError(
            f"{tool_name} not found in QNN SDK. Searched:\n"
            + "\n".join(f"  - {p}" for p in possible_paths)
        )

    def convert_onnx_to_qnn(
        self,
        onnx_path: Path,
        output_path: Path,
        input_dims: Dict[str, List[int]],
        backend: str = "cpu",
    ) -> bool:
        """
        Convert ONNX model to QNN DLC format.

        Args:
            onnx_path: Path to ONNX model
            output_path: Output DLC file path
            input_dims: Dictionary mapping input names to dimensions
            backend: QNN backend ('cpu', 'gpu', 'dsp', 'hexagon')

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"\nConverting {onnx_path.name} to QNN DLC format...")

        # Build input dimension arguments
        input_dim_args = []
        for name, dims in input_dims.items():
            dims_str = ",".join(map(str, dims))
            input_dim_args.extend(["--input_dim", f"{name}", dims_str])

        # Build command
        cmd = [
            str(self.qnn_converter),
            "--input_network", str(onnx_path),
            "--output_path", str(output_path),
        ] + input_dim_args

        # Add backend-specific flags
        if backend.lower() in ("dsp", "hexagon"):
            cmd.extend(["--target", "hexagon"])

        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info("✓ Conversion successful")
            if result.stdout:
                logger.debug(f"Output:\n{result.stdout}")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Conversion failed: {e}")
            if e.stderr:
                logger.error(f"Error output:\n{e.stderr}")
            return False

    def generate_context_binary(
        self,
        dlc_path: Path,
        output_path: Path,
        backend: str = "hexagon",
    ) -> bool:
        """
        Generate context binary for deployment.

        Args:
            dlc_path: Path to DLC file
            output_path: Output context binary path
            backend: Target backend

        Returns:
            True if successful
        """
        logger.info(f"\nGenerating context binary for {dlc_path.name}...")

        cmd = [
            str(self.qnn_context_binary_generator),
            "--model", str(dlc_path),
            "--backend", backend,
            "--output_dir", str(output_path.parent),
            "--binary_file", output_path.name,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("✓ Context binary generated")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to generate context binary: {e}")
            if e.stderr:
                logger.error(f"Error: {e.stderr}")
            return False


class QNNQuantizer:
    """
    Quantizer for QNN models (Post-Training Quantization).
    """

    def __init__(self, qnn_sdk_root: Optional[str] = None):
        self.qnn_sdk_root = Path(qnn_sdk_root or os.environ.get("QNN_SDK_ROOT"))

        if not self.qnn_sdk_root.exists():
            raise FileNotFoundError(f"QNN SDK not found at: {self.qnn_sdk_root}")

        # Note: QNN quantization tools may vary by SDK version
        # This is a placeholder - actual tool may be different
        self.quantizer_script = self.qnn_sdk_root / "bin" / "qnn-model-quantizer"

    def quantize_model(
        self,
        dlc_path: Path,
        output_path: Path,
        calibration_data_dir: Path,
        quantization_overrides: Optional[Path] = None,
    ) -> bool:
        """
        Quantize QNN model using post-training quantization.

        Args:
            dlc_path: Path to FP32 DLC model
            output_path: Output quantized DLC path
            calibration_data_dir: Directory with calibration data
            quantization_overrides: Optional JSON file with quantization settings

        Returns:
            True if successful
        """
        logger.info(f"\nQuantizing {dlc_path.name}...")
        logger.info(f"  Calibration data: {calibration_data_dir}")

        # Create calibration input list
        input_list_file = calibration_data_dir / "input_list.txt"
        if not input_list_file.exists():
            logger.warning("Creating input list from calibration data...")
            self._create_input_list(calibration_data_dir, input_list_file)

        # Quantization command (example - actual command may differ)
        cmd = [
            str(self.quantizer_script),
            "--input_dlc", str(dlc_path),
            "--output_dlc", str(output_path),
            "--input_list", str(input_list_file),
        ]

        if quantization_overrides:
            cmd.extend(["--quantization_overrides", str(quantization_overrides)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("✓ Quantization successful")
            return True

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"✗ Quantization failed: {e}")
            logger.warning(
                "\nNote: QNN quantization tools vary by SDK version.\n"
                "Please refer to your QNN SDK documentation for:\n"
                "  - qnn-quantizer\n"
                "  - qnn-model-quantizer\n"
                "  - Or alternative quantization tools"
            )
            return False

    def _create_input_list(self, calib_dir: Path, output_file: Path):
        """Create input list file for quantization."""
        npz_files = list(calib_dir.glob("*.npz"))

        with open(output_file, "w") as f:
            for npz_file in npz_files:
                f.write(f"{npz_file}\n")

        logger.info(f"  Created input list with {len(npz_files)} samples")


def convert_deepfilternet_to_qnn(
    onnx_dir: Path,
    output_dir: Path,
    backend: str = "hexagon",
    quantize: bool = False,
    calibration_dir: Optional[Path] = None,
) -> None:
    """
    Convert all DeepFilterNet ONNX models to QNN format.

    Args:
        onnx_dir: Directory containing ONNX models
        output_dir: Output directory for QNN models
        backend: QNN backend target
        quantize: Whether to perform quantization
        calibration_dir: Directory with calibration data (if quantizing)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model configurations
    models = {
        "enc": {
            "onnx": "enc.onnx",
            "inputs": {
                "feat_erb": [1, 1, 100, 32],  # [B, C, T, F] - T is dynamic
                "feat_spec": [1, 2, 100, 96],
            },
        },
        "erb_dec": {
            "onnx": "erb_dec.onnx",
            "inputs": {
                "emb": [1, 100, 256],  # [B, T, H]
                "e3": [1, 16, 100, 8],  # [B, C, T, F]
                "e2": [1, 16, 100, 8],
                "e1": [1, 16, 100, 16],
                "e0": [1, 16, 100, 32],
            },
        },
        "df_dec": {
            "onnx": "df_dec.onnx",
            "inputs": {
                "emb": [1, 100, 256],  # [B, T, H]
                "c0": [1, 16, 100, 96],  # [B, C, T, F]
            },
        },
    }

    converter = QNNConverter()

    logger.info("=" * 70)
    logger.info("CONVERTING DEEPFILTERNET MODELS TO QNN")
    logger.info("=" * 70)
    logger.info(f"Backend: {backend}")
    logger.info(f"Quantization: {'Enabled' if quantize else 'Disabled'}")

    # Convert each model
    for model_name, config in models.items():
        logger.info(f"\n[{model_name.upper()}]")
        logger.info("-" * 70)

        onnx_path = onnx_dir / config["onnx"]
        if not onnx_path.exists():
            logger.error(f"✗ ONNX model not found: {onnx_path}")
            continue

        # Convert to DLC
        dlc_path = output_dir / f"{model_name}.dlc"
        success = converter.convert_onnx_to_qnn(
            onnx_path,
            dlc_path,
            config["inputs"],
            backend=backend,
        )

        if not success:
            logger.error(f"✗ Failed to convert {model_name}")
            continue

        # Quantize if requested
        if quantize and calibration_dir:
            quantizer = QNNQuantizer()
            quantized_dlc = output_dir / f"{model_name}_quantized.dlc"

            quantizer.quantize_model(
                dlc_path,
                quantized_dlc,
                calibration_dir,
            )

        # Generate context binary for deployment
        if backend.lower() in ("hexagon", "dsp"):
            ctx_binary = output_dir / f"{model_name}.bin"
            converter.generate_context_binary(dlc_path, ctx_binary, backend)

    logger.info("\n" + "=" * 70)
    logger.info("QNN CONVERSION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nOutput directory: {output_dir}")
    logger.info("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        logger.info(f"  • {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepFilterNet ONNX models to QNN format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("onnx_dir", type=Path, help="Directory containing ONNX models")
    parser.add_argument("output_dir", type=Path, help="Output directory for QNN models")
    parser.add_argument(
        "--qnn-sdk-root",
        type=str,
        help="Path to QNN SDK root (default: $QNN_SDK_ROOT)",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "gpu", "dsp", "hexagon"],
        default="hexagon",
        help="QNN backend target (default: hexagon)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Perform post-training quantization",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        help="Directory with calibration data for quantization",
    )

    args = parser.parse_args()

    if args.quantize and not args.calibration_dir:
        logger.error("--calibration-dir required when using --quantize")
        sys.exit(1)

    try:
        convert_deepfilternet_to_qnn(
            args.onnx_dir,
            args.output_dir,
            backend=args.backend,
            quantize=args.quantize,
            calibration_dir=args.calibration_dir,
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
