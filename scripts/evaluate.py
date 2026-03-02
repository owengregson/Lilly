#!/usr/bin/env python3
"""Evaluate a trained model.

Usage:
    python scripts/evaluate.py models/run_XXX/final_model.keras
    python scripts/evaluate.py --version v2 models/v2/run_XXX/best_model.keras
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate typing model")
    parser.add_argument("model_path", type=Path, help="Path to .keras model")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--recon-samples", type=int, default=100,
                       help="V2 reconstruction samples")
    args = parser.parse_args()

    if args.version == "v2":
        _evaluate_v2(args)
    else:
        _evaluate_v1(args)


def _evaluate_v1(args):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    from lilly.core.config import TFRECORD_DIR, V1TrainConfig
    from lilly.data.pipeline import build_v1_datasets
    from lilly.evaluation.evaluator import (
        evaluate_v1,
        plot_action_confusion,
        plot_timing_distribution,
    )
    from lilly.models.lstm import compile_model

    output_dir = args.output_dir or args.model_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir or TFRECORD_DIR

    print("=" * 60)
    print("V1 Model Evaluation")
    print("=" * 60)

    model = keras.models.load_model(str(args.model_path), compile=False)
    compile_model(model)

    train_cfg = V1TrainConfig()
    _, _, test_ds, n_total = build_v1_datasets(data_dir, train_cfg, max_files=args.max_files)

    # Add sample weights for evaluation
    def add_weights(inputs, labels):
        action_labels = labels["action"]
        error_mask = tf.cast(tf.equal(action_labels, 1), tf.float32)
        return inputs, labels, {
            "timing": tf.ones_like(labels["timing"]),
            "action": tf.ones_like(tf.cast(labels["action"], tf.float32)),
            "error_char": error_mask,
        }

    test_ds = test_ds.map(add_weights, num_parallel_calls=tf.data.AUTOTUNE)

    metrics = evaluate_v1(model, test_ds)

    pred_timing = metrics.pop("_pred_timing")
    true_iki_log = metrics.pop("_true_iki_log")
    pred_action = metrics.pop("_pred_action")
    true_action = metrics.pop("_true_action")

    print("\n" + "=" * 60)
    print("Results:")
    for key, val in sorted(metrics.items()):
        if isinstance(val, float):
            print(f"  {key:30s}: {val:.4f}")
        else:
            print(f"  {key:30s}: {val}")
    print("=" * 60)

    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in metrics.items() if not k.startswith("_")}, f, indent=2)

    plot_timing_distribution(pred_timing, true_iki_log, output_dir / "timing_dist.png")
    plot_action_confusion(pred_action, true_action, output_dir / "action_confusion.png")
    print("\nDone.")


def _evaluate_v2(args):
    import json
    from lilly.core.config import V2_SEGMENT_DIR, V2ModelConfig, V2TrainConfig
    from lilly.data.pipeline import build_v2_datasets
    from lilly.evaluation.evaluator import teacher_forced_metrics, reconstruction_metrics
    from lilly.inference.preview import load_v2_model

    data_dir = args.data_dir or V2_SEGMENT_DIR
    model_cfg = V2ModelConfig()
    train_cfg = V2TrainConfig()

    print("Loading model...")
    model = load_v2_model(args.model_path)

    print("Loading data...")
    _, _, test_ds, n_total = build_v2_datasets(data_dir, model_cfg, train_cfg, max_files=args.max_files)

    print("\nTeacher-forced evaluation...")
    tf_metrics = teacher_forced_metrics(model, test_ds, train_cfg)
    for k, v in tf_metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nAutoregressive reconstruction ({args.recon_samples} samples)...")
    recon = reconstruction_metrics(model, test_ds, model_cfg, n_samples=args.recon_samples)
    for k, v in recon.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out_path = args.model_path.parent / "eval_metrics.json"
    all_metrics = {"teacher_forced": tf_metrics, "reconstruction": recon}
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    main()
