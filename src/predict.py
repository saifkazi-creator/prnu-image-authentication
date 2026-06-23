"""
predict.py
----------
Command-line inference script.  Given an image path, runs the full
forensic pipeline and outputs a prediction with a confidence score.

Usage
-----
    python src/predict.py --image path/to/image.jpg
    python src/predict.py --image path/to/image.jpg --models outputs/models/

Output
------
    Prediction  : Real Image          (or AI Generated Image)
    Confidence  : 94.37 %
    ────────────────────────────────
    Feature highlights:
      f_noise_variance     : 0.9821
      c_fft_spectral_entropy: 7.1234
      ...
"""

import argparse
import sys
from pathlib import Path

# ── Make src/ importable ──
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np

from pipeline import ForensicPipeline


def predict(image_path: str, models_dir: str = "outputs/models") -> dict:
    """Run forensic inference on a single image.

    Returns
    -------
    dict with keys: prediction (str), confidence (float), features (dict)
    """
    models_path = Path(models_dir)

    # ── Load artefacts ──
    model_pkl = models_path / "model.pkl"
    scaler_pkl = models_path / "scaler.pkl"
    cols_pkl = models_path / "feature_cols.pkl"

    for f in [model_pkl, scaler_pkl, cols_pkl]:
        if not f.exists():
            raise FileNotFoundError(
                f"Model artefact not found: {f}\n"
                "Run 'python src/train.py' first."
            )

    model = joblib.load(model_pkl)
    scaler = joblib.load(scaler_pkl)
    feature_cols = joblib.load(cols_pkl)

    # ── Run pipeline ──
    pipe = ForensicPipeline(target_size=(512, 512))
    feat_dict, original, denoised, residual = pipe.process_image(image_path)

    # ── Align feature vector with training columns ──
    feat_vector = np.array(
        [feat_dict.get(col, 0.0) for col in feature_cols],
        dtype=np.float64,
    ).reshape(1, -1)
    feat_scaled = scaler.transform(feat_vector)

    # ── Predict ──
    label_int = int(model.predict(feat_scaled)[0])
    proba = model.predict_proba(feat_scaled)[0]
    confidence = float(proba[label_int]) * 100.0

    label_str = "AI Generated Image" if label_int == 1 else "Real Image"

    return {
        "prediction": label_str,
        "label_int": label_int,
        "confidence": confidence,
        "features": feat_dict,
        "arrays": {
            "original": original,
            "denoised": denoised,
            "residual": residual,
        },
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Predict whether an image is real or AI-generated."
    )
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument(
        "--models", default="outputs/models",
        help="Directory containing model.pkl, scaler.pkl, feature_cols.pkl.",
    )
    args = parser.parse_args()

    result = predict(args.image, args.models)

    sep = "─" * 50
    print(f"\n{sep}")
    print(f"  Prediction  :  {result['prediction']}")
    print(f"  Confidence  :  {result['confidence']:.2f} %")
    print(f"{sep}")

    print("\n  Top forensic features:")
    sorted_feats = sorted(
        result["features"].items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    for name, val in sorted_feats[:10]:
        print(f"    {name:<35}: {val:>12.6f}")
    print()


if __name__ == "__main__":
    _cli()
