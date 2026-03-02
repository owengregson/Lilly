#!/usr/bin/env python3
"""Export model to TensorFlow.js format.

Usage:
    python scripts/export.py models/run_XXX/best_model.keras
    python scripts/export.py --version v2 models/v2/run_XXX/best_model.keras --quantize uint8
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export model to TF.js")
    parser.add_argument("model_path", type=Path, help="Path to .keras model file")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quantize", choices=["uint8", "uint16", "float16", "none"],
                       default="uint8")
    parser.add_argument("--keep-saved-model", action="store_true")
    args = parser.parse_args()

    if not args.model_path.exists():
        print(f"Model not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    from lilly.core.config import EXPORT_DIR, V2_EXPORT_DIR
    from lilly.export.converter import export_model

    if args.version == "v2":
        from lilly.models.transformer import (
            TransformerDecoderLayer,
            TransformerEncoderLayer,
            TypingTransformer,
        )
        custom_objects = {
            "TypingTransformer": TypingTransformer,
            "TransformerEncoderLayer": TransformerEncoderLayer,
            "TransformerDecoderLayer": TransformerDecoderLayer,
        }
        output_dir = args.output_dir or V2_EXPORT_DIR / "tfjs_model"
    else:
        from lilly.core.losses import LogNormalNLL
        custom_objects = {"LogNormalNLL": LogNormalNLL}
        output_dir = args.output_dir or EXPORT_DIR / "tfjs_model"

    print("=" * 60)
    print(f"{'V2 ' if args.version == 'v2' else ''}TensorFlow.js Model Export")
    print("=" * 60)
    print(f"  Input:       {args.model_path}")
    print(f"  Output:      {output_dir}")
    print(f"  Quantize:    {args.quantize}")
    print()

    export_model(
        args.model_path, output_dir,
        quantize=args.quantize,
        keep_saved_model=args.keep_saved_model,
        custom_objects=custom_objects,
    )

    print("\nExport complete. Copy the output directory to your extension's assets.")
    print(f"  cp -r {output_dir} <extension>/assets/model/")


if __name__ == "__main__":
    main()
