#!/usr/bin/env python3
"""Export V3 model to TensorFlow.js format.

Usage:
    python scripts/export.py models/v3/run_XXX/best_model.keras
    python scripts/export.py models/v3/run_XXX/best_model.keras --quantize float16
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export V3 model to TF.js")
    parser.add_argument("model_path", type=Path, help="Path to .keras model file")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quantize", choices=["uint8", "uint16", "float16", "none"],
                        default="uint8")
    parser.add_argument("--keep-saved-model", action="store_true")
    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Model not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    from lilly.core.config import V3_EXPORT_DIR
    from lilly.export.converter import export_model, get_v3_custom_objects

    output_dir = args.output_dir or V3_EXPORT_DIR / "tfjs_model"

    export_model(
        args.model_path, output_dir,
        quantize=args.quantize,
        keep_saved_model=args.keep_saved_model,
        custom_objects=get_v3_custom_objects(),
    )

    print("\nExport complete. Copy the output directory to your extension's assets.")
    print(f"  cp -r {output_dir} <extension>/assets/model/")


if __name__ == "__main__":
    main()
