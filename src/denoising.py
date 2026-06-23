"""
denoising.py
------------
DB8 multi-level wavelet denoising using PyWavelets.

Why DB8 for forensic denoising?
--------------------------------
Daubechies-8 (DB8) has 16 filter coefficients, giving it a long support
that captures smooth, slowly-varying signal content (the scene) while
concentrating high-frequency detail—including sensor-pattern noise—in
the detail sub-bands. Thresholding those sub-bands suppresses the scene
texture, so the difference (residual = original − denoised) is dominated
by the camera sensor's unique noise fingerprint rather than edge artifacts.
"""

from typing import Optional, Tuple

import numpy as np
import pywt


# ──────────────────────────────────────────
# WaveletDenoiser
# ──────────────────────────────────────────
class WaveletDenoiser:
    """Denoises a grayscale image with DB8 wavelet thresholding.

    Parameters
    ----------
    wavelet : str
        PyWavelets wavelet name.  Default 'db8'.
    level : int, optional
        Decomposition depth.  If None, the maximum level for the image
        size is used (capped at 5 for 512 × 512 images).
    threshold_mode : str
        'soft' or 'hard' thresholding.  Soft thresholding yields smoother
        residuals and is preferred for noise-pattern extraction.
    sigma_estimator : str
        Method for estimating noise standard deviation:
        'mad'  – median absolute deviation of the finest-scale detail (robust).
        'universal' – σ * sqrt(2 * log N) universal threshold.
    """

    def __init__(
        self,
        wavelet: str = "db8",
        level: Optional[int] = None,
        threshold_mode: str = "soft",
        sigma_estimator: str = "mad",
    ) -> None:
        self.wavelet = wavelet
        self.level = level
        self.threshold_mode = threshold_mode
        self.sigma_estimator = sigma_estimator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """Return a denoised version of *image*.

        Parameters
        ----------
        image : np.ndarray  float64, shape (H, W), values in [0, 1]

        Returns
        -------
        np.ndarray  float64, same shape, values clipped to [0, 1]
        """
        if image.ndim != 2:
            raise ValueError("Expected a 2-D (grayscale) array.")

        # ── Determine decomposition level ──
        max_level = pywt.dwt_max_level(min(image.shape), self.wavelet)
        level = min(self.level or max_level, 5)

        # ── Forward wavelet transform ──
        coeffs = pywt.wavedec2(image, wavelet=self.wavelet, level=level)

        # ── Threshold each detail sub-band ──
        coeffs_thresh = [coeffs[0]]  # keep LL approximation intact
        for detail_tuple in coeffs[1:]:
            threshold = self._compute_threshold(detail_tuple)
            thresholded = tuple(
                pywt.threshold(sub, threshold, mode=self.threshold_mode)
                for sub in detail_tuple
            )
            coeffs_thresh.append(thresholded)

        # ── Reconstruct ──
        denoised = pywt.waverec2(coeffs_thresh, wavelet=self.wavelet)

        # ── Crop / pad to original shape and clip ──
        h, w = image.shape
        denoised = denoised[:h, :w]
        return np.clip(denoised, 0.0, 1.0)

    def get_wavelet_coefficients(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, list]:
        """Return raw wavelet coefficients (LL, then detail tuples).

        Useful for Feature Group E (wavelet statistics).
        """
        max_level = pywt.dwt_max_level(min(image.shape), self.wavelet)
        level = min(self.level or max_level, 5)
        coeffs = pywt.wavedec2(image, wavelet=self.wavelet, level=level)
        return coeffs[0], list(coeffs[1:])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_threshold(self, detail_tuple: tuple) -> float:
        """Estimate a per-level threshold from the detail sub-bands."""
        # Concatenate all detail coefficients at this level
        all_detail = np.concatenate([sub.ravel() for sub in detail_tuple])

        if self.sigma_estimator == "mad":
            # Robust noise-σ estimate via the finest-scale HH band
            sigma = np.median(np.abs(all_detail)) / 0.6745
        else:
            sigma = np.std(all_detail)

        # Universal threshold  T = σ √(2 ln N)
        n = all_detail.size
        threshold = sigma * np.sqrt(2.0 * np.log(max(n, 2)))
        return float(threshold)
