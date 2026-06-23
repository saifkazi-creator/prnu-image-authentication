"""
train.py
--------
Builds the feature CSV, trains five classifiers, evaluates them,
and saves the best model + scaler to disk.

Usage
-----
    python src/train.py --dataset dataset/ --output outputs/

Arguments
---------
--dataset   Root folder containing real/ and ai/ sub-directories.
--output    Output directory for CSV, models, and figures.
--seed      Random seed (default 42).
--max-imgs  Cap images per class (useful for quick experiments).
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# ── Make src/ importable when called from project root ──
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline import ForensicPipeline
from evaluate import Evaluator

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
# Classifiers
# ────────────────────────────────────────────────────────────────────────────

def build_classifiers(seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=seed,
            verbosity=0,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=seed,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            random_state=seed,
            verbose=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        ),
    }


# ────────────────────────────────────────────────────────────────────────────
# Feature-CSV builder
# ────────────────────────────────────────────────────────────────────────────

def build_feature_csv(
    dataset_dir: str,
    output_dir: str,
    residual_dir: str,
    max_images: int | None,
) -> pd.DataFrame:
    """Extract features for all images and return a DataFrame."""
    pipe = ForensicPipeline(
        target_size=(512, 512),
        residual_output_dir=residual_dir,
    )

    rows = []
    for label_int, sub in [(0, "real"), (1, "ai")]:
        folder = Path(dataset_dir) / sub
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing subfolder: {folder}")

        exts = {".jpg", ".jpeg", ".png"}
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
        if max_images:
            paths = paths[:max_images]

        print(f"\n[INFO] Processing '{sub}' ({len(paths)} images)...")
        for p in tqdm(paths, unit="img"):
            try:
                feat_dict, *_ = pipe.process_image(str(p))
                feat_dict["label"] = label_int
                feat_dict["filename"] = p.name
                rows.append(feat_dict)
            except Exception as exc:
                print(f"  [WARN] {p.name}: {exc}")

    df = pd.DataFrame(rows)
    csv_path = Path(output_dir) / "features" / "features.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Feature CSV saved → {csv_path}  ({len(df)} rows)")
    return df


# ────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

    seed = args.seed
    np.random.seed(seed)
    output_dir = Path(args.output)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    # ── 1. Build / load feature CSV ──
    csv_path = output_dir / "features" / "features.csv"
    if csv_path.exists() and not args.rebuild_csv:
        print(f"[INFO] Loading existing feature CSV: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = build_feature_csv(
            dataset_dir=args.dataset,
            output_dir=str(output_dir),
            residual_dir=str(output_dir / "residuals"),
            max_images=args.max_imgs,
        )

    # ── 2. Prepare X, y ──
    drop_cols = {"label", "filename"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # ── Clean NaN / Inf values produced by degenerate images ──
    before = len(df)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    bad_rows = df[feature_cols].isnull().any(axis=1)
    if bad_rows.sum() > 0:
        print(f"[WARN] Dropping {bad_rows.sum()} rows with NaN/Inf features "
              f"(out of {before}).")
        df = df[~bad_rows].reset_index(drop=True)

    X = df[feature_cols].values.astype(np.float64)
    y = df["label"].values.astype(np.int32)

    print(f"\n[INFO] Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"       Class distribution: real={int((y==0).sum())}, ai={int((y==1).sum())}")

    # ── 3. EDA plots ──
    evaluator = Evaluator(output_dir=str(output_dir / "figures"))
    pass  # skipping EDA plots

    # ── 4. Train / test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # ── 5. Scale ──
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── 6. Train all classifiers + cross-validate ──
    classifiers = build_classifiers(seed)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    results = {}
    best_name, best_f1, best_model = None, -1.0, None

    print("\n" + "=" * 60)
    print("  CROSS-VALIDATION RESULTS (5-Fold Stratified)")
    print("=" * 60)

    for name, clf in classifiers.items():
        cv_scores = cross_val_score(
            clf, X_train_s, y_train,
            cv=kfold, scoring="f1", n_jobs=-1
        )
        print(f"  {name:<15}  F1 = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Final fit on all training data
        clf.fit(X_train_s, y_train)
        metrics = evaluator.evaluate(clf, X_test_s, y_test, name)
        results[name] = metrics

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name
            best_model = clf

    # ── 7. Save best model + scaler ──
    model_dir = output_dir / "models"
    joblib.dump(best_model, model_dir / "model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    joblib.dump(feature_cols, model_dir / "feature_cols.pkl")
    print(f"\n[INFO] Best model: {best_name}  (F1 = {best_f1:.4f})")
    print(f"[INFO] Saved → {model_dir / 'model.pkl'}")
    print(f"[INFO] Saved → {model_dir / 'scaler.pkl'}")

    # ── 8. Feature importance ──
    evaluator.plot_feature_importance(
        best_model, best_name, feature_cols,
        classifiers.get("RandomForest"),
        classifiers.get("XGBoost"),
        X_train_s, y_train,
    )

    # ── 9. Summary table ──
    print("\n" + "=" * 60)
    print("  HOLD-OUT TEST SET RESULTS")
    print("=" * 60)
    summary = pd.DataFrame(results).T
    print(summary.to_string())
    summary.to_csv(output_dir / "figures" / "results_summary.csv")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AI-image-detection classifiers using PRNU forensics."
    )
    parser.add_argument(
        "--dataset", default="dataset/",
        help="Root dataset folder (must contain real/ and ai/ sub-dirs).",
    )
    parser.add_argument(
        "--output", default="outputs/",
        help="Directory for outputs (CSV, models, figures).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Global random seed."
    )
    parser.add_argument(
        "--max-imgs", type=int, default=None,
        help="Maximum images per class (for quick runs).",
    )
    parser.add_argument(
        "--rebuild-csv", action="store_true",
        help="Re-extract features even if features.csv already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())