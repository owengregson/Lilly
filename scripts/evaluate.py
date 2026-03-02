#!/usr/bin/env python3
"""Evaluate a trained V3 model.

Usage:
    python scripts/evaluate.py models/v3/run_XXX/best_model.keras
    python scripts/evaluate.py models/v3/run_XXX/best_model.keras --tier 2
    python scripts/evaluate.py models/v3/run_XXX/best_model.keras --tier 3 --n-samples 500
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate V3 typing model")
    parser.add_argument("model_path", type=Path, help="Path to .keras model")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3],
                        help="Evaluation tier (1=point, 2=distributional, 3=realism)")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization plots (Tier 1 only)")
    args = parser.parse_args()

    import numpy as np
    from tensorflow import keras

    from lilly.core.config import V3_SEGMENT_DIR, V3ModelConfig, V3TrainConfig
    from lilly.data.pipeline import build_v3_datasets
    from lilly.export.converter import get_v3_custom_objects

    model_cfg = V3ModelConfig()
    train_cfg = V3TrainConfig()
    data_dir = args.data_dir or V3_SEGMENT_DIR
    output_dir = args.output_dir or args.model_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"V3 Model Evaluation — Tier {args.tier}")
    print("=" * 60)

    # Load model
    print(f"Loading model: {args.model_path}")
    model = keras.models.load_model(
        str(args.model_path), compile=False,
        custom_objects=get_v3_custom_objects(),
    )

    # Load dataset
    print(f"Loading data from: {data_dir}")
    _, _, test_ds, n_total = build_v3_datasets(
        data_dir, model_cfg, train_cfg, max_files=args.max_files,
    )
    print(f"Total samples: {n_total}")

    metrics = {}

    if args.tier >= 1:
        print("\n--- Tier 1: Point Metrics ---")
        from lilly.evaluation.metrics import compute_tier1_metrics
        from lilly.training.losses import V3LossConfig

        t1 = compute_tier1_metrics(model, test_ds, V3LossConfig())
        metrics["tier1"] = t1
        for k, v in sorted(t1.items()):
            print(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")

    if args.tier >= 2:
        print("\n--- Tier 2: Distributional Metrics ---")
        from lilly.evaluation.distributional import compute_tier2_metrics

        t2 = compute_tier2_metrics(model, test_ds, model_cfg, n_samples=args.n_samples)
        metrics["tier2"] = t2
        for k, v in sorted(t2.items()):
            print(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")

    if args.tier >= 3:
        print("\n--- Tier 3: Realism Metrics ---")
        from lilly.evaluation.realism import check_style_consistency, compute_realism_score

        t3 = compute_realism_score(model, test_ds, model_cfg, n_samples=args.n_samples)
        style = check_style_consistency(model, model_cfg)
        t3.update(style)
        metrics["tier3"] = t3
        for k, v in sorted(t3.items()):
            print(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k:30s}: {v}")

    if args.visualize:
        print("\n--- Generating Visualizations ---")
        from lilly.evaluation.visualization import plot_action_confusion, plot_mdn_components

        # Collect predictions for visualization
        all_true_actions = []
        all_pred_actions = []
        all_real_ikis = []

        for batch_inputs, batch_labels in test_ds:
            outputs = model(batch_inputs, training=False)
            mask = batch_labels["label_mask"].numpy()
            action_probs = outputs["action_probs"].numpy()
            pred_actions = np.argmax(action_probs, axis=-1)
            true_actions = batch_labels["action_labels"].numpy()
            delays = batch_labels["delay_labels"].numpy()

            for b in range(len(mask)):
                for t in range(mask.shape[1]):
                    if mask[b, t] > 0:
                        all_true_actions.append(true_actions[b, t])
                        all_pred_actions.append(pred_actions[b, t])
                        all_real_ikis.append(delays[b, t])

        all_true_actions = np.array(all_true_actions)
        all_pred_actions = np.array(all_pred_actions)
        all_real_ikis = np.array(all_real_ikis)

        plot_action_confusion(
            all_pred_actions, all_true_actions,
            output_dir / "action_confusion.png",
        )
        plot_mdn_components(model, output_dir / "mdn_components.png")
        print("Visualizations saved.")

    # Save metrics
    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
