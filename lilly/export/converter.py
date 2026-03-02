"""Unified model export to TensorFlow.js format.

Handles both V1 (LSTM) and V2 (Transformer) models through a single
pipeline: Keras (.keras) -> TF SavedModel -> TF.js graph model.

Supports uint8, uint16, float16 quantization for Chrome extension deployment.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from tensorflow import keras


def get_v3_custom_objects() -> dict:
    """Return V3 custom objects dict for model deserialization."""
    from lilly.models.components import (
        ActionGate,
        ErrorCharHead,
        FiLMModulation,
        MDNHead,
    )
    from lilly.models.typing_model import TypingTransformerV3
    return {
        "TypingTransformerV3": TypingTransformerV3,
        "FiLMModulation": FiLMModulation,
        "MDNHead": MDNHead,
        "ActionGate": ActionGate,
        "ErrorCharHead": ErrorCharHead,
    }


def export_saved_model(
    keras_path: Path,
    saved_model_dir: Path,
    custom_objects: dict | None = None,
) -> None:
    """Load a Keras model and export as TF SavedModel.

    This is the intermediate step before TF.js conversion.

    Parameters
    ----------
    keras_path : Path
        Path to the ``.keras`` model file.
    saved_model_dir : Path
        Directory where the SavedModel will be written.
    custom_objects : dict or None
        Optional dict of custom objects needed for deserialization
        (e.g. ``{"TypingTransformer": TypingTransformer}`` for V2 models).
    """
    print(f"Loading Keras model: {keras_path}")
    model = keras.models.load_model(
        str(keras_path), compile=False, custom_objects=custom_objects
    )
    model.summary()

    saved_model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting SavedModel to: {saved_model_dir}")
    model.export(str(saved_model_dir))
    print("SavedModel export complete.")


def convert_to_tfjs(
    saved_model_dir: Path,
    output_dir: Path,
    quantize: str = "uint8",
) -> None:
    """Convert a TF SavedModel to TF.js graph model format.

    Runs ``tensorflowjs_converter`` as a subprocess with the specified
    quantization level.

    Parameters
    ----------
    saved_model_dir : Path
        Path to the SavedModel directory.
    output_dir : Path
        Output directory for the TF.js model files.
    quantize : str
        Quantization type: ``"uint8"``, ``"uint16"``, ``"float16"``,
        or ``"none"`` for no quantization.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "tensorflowjs_converter",
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
        "--signature_name=serving_default",
        "--saved_model_tags=serve",
    ]

    if quantize == "uint8":
        cmd.append("--quantize_uint8")
    elif quantize == "uint16":
        cmd.append("--quantize_uint16")
    elif quantize == "float16":
        cmd.append("--quantize_float16")
    # "none" = no quantization flag

    cmd.extend([str(saved_model_dir), str(output_dir)])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"TFJS conversion failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    print(f"TF.js model exported to: {output_dir}")


def report_size(output_dir: Path) -> None:
    """Report the total size of exported TF.js model files.

    Prints a table of all files in the output directory with sizes,
    then reads ``model.json`` for format and weight manifest information.

    Parameters
    ----------
    output_dir : Path
        Directory containing the exported TF.js model files.
    """
    total = 0
    files = sorted(output_dir.iterdir())

    print("\nExported files:")
    for f in files:
        size = f.stat().st_size
        total += size
        print(f"  {f.name:40s} {size:>10,} bytes ({size / 1024:.1f} KB)")

    print(
        f"\n  Total: {total:,} bytes ({total / 1024:.1f} KB, "
        f"{total / (1024 * 1024):.2f} MB)"
    )

    # Read model.json for architecture info
    model_json = output_dir / "model.json"
    if model_json.exists():
        with open(model_json) as f:
            meta = json.load(f)
        if "modelTopology" in meta:
            print(f"  Format: {meta.get('format', 'unknown')}")
        if "weightsManifest" in meta:
            n_weights = sum(
                len(group.get("weights", []))
                for group in meta["weightsManifest"]
            )
            print(f"  Weight groups: {n_weights}")


def export_model(
    keras_path: Path,
    output_dir: Path,
    quantize: str = "uint8",
    keep_saved_model: bool = False,
    custom_objects: dict | None = None,
) -> None:
    """High-level export: Keras -> SavedModel -> TF.js with size report.

    Orchestrates the full export pipeline for either V1 or V2 models.

    Parameters
    ----------
    keras_path : Path
        Path to the ``.keras`` model file.
    output_dir : Path
        Output directory for the TF.js model files.
    quantize : str
        Quantization type: ``"uint8"``, ``"uint16"``, ``"float16"``,
        or ``"none"``.
    keep_saved_model : bool
        If True, keep the intermediate SavedModel directory. Otherwise
        it is deleted after conversion.
    custom_objects : dict or None
        Optional dict of custom objects needed for loading the Keras model
        (e.g. V2 transformer classes).
    """
    saved_model_dir = output_dir.parent / "_saved_model_tmp"

    print("=" * 60)
    print("TensorFlow.js Model Export")
    print("=" * 60)
    print(f"  Input:       {keras_path}")
    print(f"  Output:      {output_dir}")
    print(f"  Quantize:    {quantize}")
    print()

    # Step 1: Keras -> SavedModel
    export_saved_model(keras_path, saved_model_dir, custom_objects=custom_objects)
    print()

    # Step 2: SavedModel -> TF.js
    convert_to_tfjs(saved_model_dir, output_dir, quantize)

    # Cleanup intermediate SavedModel
    if not keep_saved_model and saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)
        print("Cleaned up intermediate SavedModel.")

    # Report exported file sizes
    report_size(output_dir)
