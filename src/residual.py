"""
residual.py
-----------
Extracts the noise residual from an image after wavelet denoising.

    Residual = Original − Denoised

Why the residual distinguishes real cameras from AI generators?
----------------------------------------------------------------
Real camera images carry a subtle, spatially correlated noise pattern
impressed by the sensor's photo-response non-uniformity (PRNU).  This
pattern persists across images from the same device and makes the residual
statistically non-white.

AI generators (diffusion, GAN, auto-regressive) synthesise pixels from
a learned distribution rather than a physical sensor.  Their residuals
are structurally different:
  - Diffusion models leave characteristic spectral signatures from the
    denoising U-Net and scheduler steps.
  - GANs imprint periodic grid patterns from transposed-convolution layers.
  - Both tend to produce residuals with much weaker spatial correlation
    than real PRNU but with different frequency-domain signatures.

These differences are precisely what the downstream feature extractor
and classifier exploit.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ──────────────────────────────────────────
# ResidualExtractor
# ──────────────────────────────────────────
class ResidualExtractor:
    """Computes and (optionally) saves noise residuals.

    Parameters
    ----------
    output_dir : str, optional
        Directory where residual PNG files are saved.
        Pass None to skip saving.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        original: np.ndarray,
        denoised: np.ndarray,
        image_name: Optional[str] = None,
    ) -> np.ndarray:
        """Compute the normalised noise residual.

        Parameters
        ----------
        original : np.ndarray  float64, shape (H, W), values in [0, 1]
        denoised : np.ndarray  float64, same shape
        image_name : str, optional
            Stem used when saving the residual to disk.

        Returns
        -------
        residual : np.ndarray  float64, shape (H, W)
            Zero-mean, unit-variance noise residual.
        """
        if original.shape != denoised.shape:
            raise ValueError(
                f"Shape mismatch: original {original.shape} vs "
                f"denoised {denoised.shape}."
            )

        raw_residual = original - denoised

        # Z-score normalisation so variance reflects signal content, not scale.
        residual = self._normalise(raw_residual)

        if self.output_dir is not None and image_name is not None:
            self._save_residual(residual, image_name)

        return residual

    def extract_raw(
        self, original: np.ndarray, denoised: np.ndarray
    ) -> np.ndarray:
        """Return the un-normalised residual (useful for visualisation)."""
        return original - denoised

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(residual: np.ndarray) -> np.ndarray:
        """Zero-mean, unit-std normalisation.  Falls back to zero-mean if
        the standard deviation is below a numerical floor."""
        mu = residual.mean()
        sigma = residual.std()
        if sigma < 1e-10:
            return residual - mu
        return (residual - mu) / sigma

    def _save_residual(self, residual: np.ndarray, name: str) -> None:
        """Scale residual to [0, 255] and save as PNG."""
        # Import here to keep top-level import light
        from PIL import Image as PILImage

        # Map to 8-bit for storage
        rmin, rmax = residual.min(), residual.max()
        if rmax - rmin < 1e-10:
            vis = np.zeros_like(residual, dtype=np.uint8)
        else:
            vis = ((residual - rmin) / (rmax - rmin) * 255).astype(np.uint8)

        stem = Path(name).stem
        out_path = self.output_dir / f"{stem}_residual.png"
        PILImage.fromarray(vis).save(str(out_path))
