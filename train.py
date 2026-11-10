"""
Standalone training script — run once to train and save the model.
Usage:  python train.py [--csv path/to/data.csv]
"""
import argparse
import sys
from model_utils import train_model

def main():
    parser = argparse.ArgumentParser(description="Train PMGSY Scheme Classifier")
    parser.add_argument("--csv", default=None, help="Path to PMGSY CSV file (optional)")
    args = parser.parse_args()

    print("=" * 60)
    print("PMGSY Scheme Classification — Model Training")
    print("=" * 60)

    print("\n[1/3] Loading and preprocessing data…")
    results, artifacts = train_model(csv_path=args.csv)

    print("\n[2/3] Evaluation Results:")
    print(f"  Accuracy        : {results['accuracy']:.4f}  ({results['accuracy']:.2%})")
    print(f"  Weighted F1     : {results['f1_weighted']:.4f}")
    print(f"  Classes         : {results['scheme_labels']}")

    print("\n  Confusion Matrix:")
    import numpy as np
    labels = results["scheme_labels"]
    cm = results["confusion_matrix"]
    header = "         " + "  ".join(f"{l[:7]:>8}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = "  ".join(f"{v:>8}" for v in cm[i])
        print(f"  {row_label:<8} {row}")

    print("\n  Top-5 Feature Importances:")
    for feat, imp in results["feature_importances"][:5]:
        bar = "#" * int(imp * 50)
        print(f"  {feat:<25} {imp:.4f}  {bar}")

    print(f"\n[3/3] Model saved -> {results['artifacts_path']}")
    print("\n[OK] Training complete. Launch the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
