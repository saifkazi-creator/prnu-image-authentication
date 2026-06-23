"""
evaluate.py
-----------
Evaluation utilities: metrics, confusion matrix, ROC curve, EDA
plots, feature-importance plots, and SHAP analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)


LABEL_NAMES = {0: "Real", 1: "AI"}


class Evaluator:
    """Centralises all evaluation and visualisation logic.

    Parameters
    ----------
    output_dir : str
        Directory where figures are saved.
    """

    def __init__(self, output_dir: str = "outputs/figures") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "Model",
    ) -> Dict[str, float]:
        """Compute metrics and generate evaluation plots.

        Returns
        -------
        dict with accuracy, precision, recall, f1, roc_auc.
        """
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else np.zeros_like(y_pred, dtype=float)
        )

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }

        print(f"\n── {model_name} ──")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Real", "AI"]))

        self._plot_confusion_matrix(y_test, y_pred, model_name)
        self._plot_roc_curve(y_test, y_prob, model_name)

        return metrics

    # ------------------------------------------------------------------
    # EDA plots
    # ------------------------------------------------------------------

    def plot_eda(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        y: np.ndarray,
    ) -> None:
        """Generate and save EDA figures."""
        print("\n[INFO] Generating EDA plots...")
        X = df[feature_cols].values

        self._plot_feature_distributions(df, feature_cols)
        self._plot_correlation_matrix(df, feature_cols)
        self._plot_pca(X, y)
        self._plot_tsne(X, y)
        self._plot_class_comparison(df, feature_cols, y)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def plot_feature_importance(
        self,
        best_model: Any,
        best_name: str,
        feature_names: List[str],
        rf_model: Optional[Any] = None,
        xgb_model: Optional[Any] = None,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
    ) -> None:
        """Plot RF importance, XGB importance, and SHAP values."""
        print("\n[INFO] Generating feature importance plots...")

        if rf_model is not None and hasattr(rf_model, "feature_importances_"):
            self._plot_importances(
                rf_model.feature_importances_, feature_names, "RandomForest"
            )

        if xgb_model is not None and hasattr(xgb_model, "feature_importances_"):
            self._plot_importances(
                xgb_model.feature_importances_, feature_names, "XGBoost"
            )

        if X_train is not None and y_train is not None:
            self._plot_shap(best_model, best_name, X_train, feature_names)

    # ------------------------------------------------------------------
    # Private plot helpers
    # ------------------------------------------------------------------

    def _plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, name: str
    ) -> None:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Real", "AI"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix – {name}")
        fig.tight_layout()
        fig.savefig(self.output_dir / f"cm_{name}.png", dpi=120)
        plt.close(fig)

    def _plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        name: str,
    ) -> None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve – {name}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / f"roc_{name}.png", dpi=120)
        plt.close(fig)

    def _plot_feature_distributions(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> None:
        n_cols = 5
        n_rows = (len(feature_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))
        axes_flat = axes.ravel()
        for i, col in enumerate(feature_cols):
            for lbl, grp in df.groupby("label"):
                vals = grp[col].replace([np.inf,-np.inf],np.nan).dropna().values
                if len(vals)<2: continue
                cmin,cmax=float(vals.min()),float(vals.max())
                if abs(cmax-cmin)<1e-10: axes_flat[i].axvline(cmin,alpha=0.6,label=LABEL_NAMES[lbl]);continue
                axes_flat[i].hist(vals,bins=min(30,len(set(vals))),alpha=0.5,label=LABEL_NAMES[lbl],density=True,range=(cmin,cmax))
            axes_flat[i].set_title(col, fontsize=7)
            axes_flat[i].legend(fontsize=6)
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle("Feature Distributions: Real vs AI", y=1.01, fontsize=12)
        fig.tight_layout()
        fig.savefig(self.output_dir / "feature_distributions.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    def _plot_correlation_matrix(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> None:
        corr = df[feature_cols].corr()
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, cmap="coolwarm", center=0,
            linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.6},
        )
        ax.set_title("Feature Correlation Matrix", fontsize=14)
        fig.tight_layout()
        fig.savefig(self.output_dir / "correlation_matrix.png", dpi=100)
        plt.close(fig)

    def _plot_pca(self, X: np.ndarray, y: np.ndarray) -> None:
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
        fig, ax = plt.subplots(figsize=(6, 5))
        for lbl, name in LABEL_NAMES.items():
            mask = y == lbl
            ax.scatter(X2[mask, 0], X2[mask, 1], alpha=0.5, s=15, label=name)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title("PCA – Real vs AI Residual Features")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "pca.png", dpi=120)
        plt.close(fig)

    def _plot_tsne(self, X: np.ndarray, y: np.ndarray) -> None:
        # Limit to 1000 samples for speed
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=min(1000, len(X)), replace=False)
        Xs, ys = X[idx], y[idx]

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
        X2 = tsne.fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(6, 5))
        for lbl, name in LABEL_NAMES.items():
            mask = ys == lbl
            ax.scatter(X2[mask, 0], X2[mask, 1], alpha=0.5, s=15, label=name)
        ax.set_title("t-SNE – Real vs AI Residual Features")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.output_dir / "tsne.png", dpi=120)
        plt.close(fig)

    def _plot_class_comparison(
        self, df: pd.DataFrame, feature_cols: List[str], y: np.ndarray
    ) -> None:
        means = (
            df.groupby("label")[feature_cols]
            .mean()
            .rename(index=LABEL_NAMES)
        )
        top_features = (
            (means.loc["AI"] - means.loc["Real"])
            .abs()
            .nlargest(20)
            .index.tolist()
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        means[top_features].T.plot(kind="bar", ax=ax, alpha=0.8)
        ax.set_title("Top-20 Feature Means: Real vs AI")
        ax.set_ylabel("Mean value (scaled)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Class")
        fig.tight_layout()
        fig.savefig(self.output_dir / "class_comparison.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    def _plot_importances(
        self,
        importances: np.ndarray,
        feature_names: List[str],
        name: str,
        top_n: int = 20,
    ) -> None:
        idx = np.argsort(importances)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            [feature_names[i] for i in reversed(idx)],
            importances[idx[::-1]],
            color="steelblue",
        )
        ax.set_title(f"Feature Importance – {name} (Top {top_n})")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(self.output_dir / f"importance_{name}.png", dpi=120)
        plt.close(fig)

    def _plot_shap(
        self,
        model: Any,
        model_name: str,
        X_train: np.ndarray,
        feature_names: List[str],
        max_samples: int = 200,
    ) -> None:
        try:
            import shap

            # Subsample for speed
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_train), size=min(max_samples, len(X_train)), replace=False)
            X_sample = X_train[idx]

            # Choose explainer based on model type
            model_type = type(model).__name__
            if model_type in ("RandomForestClassifier", "XGBClassifier", "LGBMClassifier"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                # For multi-output RF, take class-1 slice
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
            else:
                explainer = shap.KernelExplainer(
                    model.predict_proba, shap.sample(X_sample, 50)
                )
                shap_values = explainer.shap_values(X_sample, nsamples=100)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

            fig, ax = plt.subplots(figsize=(8, 7))
            shap.summary_plot(
                shap_values, X_sample,
                feature_names=feature_names,
                show=False, plot_type="bar",
            )
            plt.title(f"SHAP Feature Importance – {model_name}")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"shap_{model_name}.png", dpi=120, bbox_inches="tight")
            plt.close("all")

        except Exception as exc:
            print(f"[WARN] SHAP plot failed for {model_name}: {exc}")