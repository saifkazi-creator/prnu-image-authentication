"""
features.py
-----------
Extracts a rich, multi-group forensic feature vector from a noise residual.

Feature Groups and their forensic rationale:
=============================================

Group A – Residual Statistics
  Basic statistical moments of the residual pixel values.
  Real camera residuals have near-zero mean (PRNU is zero-mean) but
  measurable kurtosis/skewness because the noise is signal-dependent.
  AI residuals differ: diffusion-model residuals tend to be more
  Gaussian (higher kurtosis ~3) while GAN residuals are platykurtic.

Group B – Energy Features
  Total and average squared residual energy.  Real sensors inject a
  consistent amount of PRNU energy; AI generators usually produce
  either very low or patterned energy.

Group C – FFT Features
  The power spectrum of the residual.  Real PRNU has a roughly flat
  (white-noise-like) spectrum after denoising.  AI generators leave
  spectral peaks: GANs at spatial frequencies matching their stride,
  diffusion models with systematic low-frequency energy from the
  denoising U-Net.

Group D – Autocorrelation Features
  Spatial autocorrelation of the residual.  Genuine sensor noise is
  nearly spatially independent (fast autocorrelation decay).  AI
  residuals show long-range spatial structure from convolutional
  layers and attention mechanisms.

Group E – Wavelet Sub-band Statistics
  Statistics of LL, LH, HL, HH sub-bands from a single-level
  decomposition of the residual.  Real images concentrate noise in
  HH; AI generators sometimes leak energy into LL because of the
  coarse-to-fine nature of their sampling processes.

Group F – PRNU-style Features
  Noise variance, energy, local consistency, and residual correlation
  mimic the classical PRNU fingerprint pipeline.  These are the most
  directly interpretable forensic features.
"""

from typing import Dict, List, Optional

import numpy as np
import pywt
from scipy import stats
from scipy.signal import correlate2d


# ──────────────────────────────────────────
# Constants
# ──────────────────────────────────────────
WAVELET: str = "db8"
LOCAL_BLOCK_SIZE: int = 32   # for local-noise-consistency (Group F)
EPS: float = 1e-10


# ──────────────────────────────────────────
# FeatureExtractor
# ──────────────────────────────────────────
class FeatureExtractor:
    """Computes a flat forensic feature vector from a noise residual.

    Parameters
    ----------
    local_block_size : int
        Side length (pixels) of non-overlapping blocks used for
        local-noise-consistency estimation (Group F).
    """

    def __init__(self, local_block_size: int = LOCAL_BLOCK_SIZE) -> None:
        self.local_block_size = local_block_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, residual: np.ndarray) -> Dict[str, float]:
        """Compute all feature groups and return a flat dict.

        Parameters
        ----------
        residual : np.ndarray  float64, shape (H, W)

        Returns
        -------
        dict : {feature_name: value}
        """
        if residual.ndim != 2:
            raise ValueError("Residual must be 2-D (grayscale).")

        features: Dict[str, float] = {}
        features.update(self._group_a_statistics(residual))
        features.update(self._group_b_energy(residual))
        features.update(self._group_c_fft(residual))
        features.update(self._group_d_autocorrelation(residual))
        features.update(self._group_e_wavelet(residual))
        features.update(self._group_f_prnu(residual))
        return features

    def feature_names(self) -> List[str]:
        """Return feature names in the same order as :meth:`extract`."""
        dummy = np.random.default_rng(0).standard_normal((64, 64))
        return list(self.extract(dummy).keys())

    # ------------------------------------------------------------------
    # Group A – Residual Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _group_a_statistics(r: np.ndarray) -> Dict[str, float]:
        flat = r.ravel()
        entropy_val = FeatureExtractor._shannon_entropy(flat)
        return {
            "a_mean": float(np.mean(flat)),
            "a_variance": float(np.var(flat)),
            "a_std": float(np.std(flat)),
            "a_rms": float(np.sqrt(np.mean(flat ** 2))),
            "a_skewness": float(stats.skew(flat)),
            "a_kurtosis": float(stats.kurtosis(flat)),
            "a_entropy": float(entropy_val),
        }

    # ------------------------------------------------------------------
    # Group B – Energy Features
    # ------------------------------------------------------------------

    @staticmethod
    def _group_b_energy(r: np.ndarray) -> Dict[str, float]:
        energy = float(np.sum(r ** 2))
        return {
            "b_total_energy": energy,
            "b_average_energy": energy / r.size,
        }

    # ------------------------------------------------------------------
    # Group C – FFT Features
    # ------------------------------------------------------------------

    @staticmethod
    def _group_c_fft(r: np.ndarray) -> Dict[str, float]:
        fft2 = np.fft.fft2(r)
        magnitude = np.abs(np.fft.fftshift(fft2))

        flat_mag = magnitude.ravel()
        h, w = magnitude.shape
        cx, cy = h // 2, w // 2

        # Flat spectral entropy
        spec_entropy = FeatureExtractor._shannon_entropy(flat_mag)

        # High- vs low-frequency split at Nyquist/4
        radius = np.sqrt(
            (np.arange(h)[:, None] - cx) ** 2
            + (np.arange(w)[None, :] - cy) ** 2
        )
        cutoff = min(cx, cy) // 2
        low_mask = radius <= cutoff
        high_mask = radius > cutoff

        total_energy = float(np.sum(magnitude ** 2)) + EPS
        low_energy = float(np.sum(magnitude[low_mask] ** 2))
        high_energy = float(np.sum(magnitude[high_mask] ** 2))

        return {
            "c_fft_mean_mag": float(np.mean(flat_mag)),
            "c_fft_var_mag": float(np.var(flat_mag)),
            "c_fft_spectral_entropy": float(spec_entropy),
            "c_fft_peak_mag": float(np.max(flat_mag)),
            "c_fft_high_freq_ratio": high_energy / total_energy,
            "c_fft_low_freq_ratio": low_energy / total_energy,
        }

    # ------------------------------------------------------------------
    # Group D – Autocorrelation Features
    # ------------------------------------------------------------------

    @staticmethod
    def _group_d_autocorrelation(r: np.ndarray) -> Dict[str, float]:
        # Downsample to 64 × 64 to keep correlation tractable
        from skimage.transform import resize as sk_resize
        r_small = sk_resize(r, (64, 64), anti_aliasing=True)

        acorr = correlate2d(r_small, r_small, mode="full", boundary="wrap")
        # Exclude the central peak (zero-lag autocorrelation)
        cy, cx = np.array(acorr.shape) // 2
        mask = np.ones(acorr.shape, dtype=bool)
        mask[cy - 2 : cy + 3, cx - 2 : cx + 3] = False

        off_peak = acorr[mask]
        return {
            "d_acorr_max": float(np.max(np.abs(off_peak))),
            "d_acorr_mean": float(np.mean(np.abs(off_peak))),
            "d_acorr_var": float(np.var(off_peak)),
        }

    # ------------------------------------------------------------------
    # Group E – Wavelet Sub-band Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _group_e_wavelet(r: np.ndarray) -> Dict[str, float]:
        coeffs = pywt.dwt2(r, wavelet=WAVELET)
        ll = coeffs[0]
        lh, hl, hh = coeffs[1]

        feats: Dict[str, float] = {}
        for name, sub in [("ll", ll), ("lh", lh), ("hl", hl), ("hh", hh)]:
            flat = sub.ravel()
            feats[f"e_{name}_mean"] = float(np.mean(flat))
            feats[f"e_{name}_var"] = float(np.var(flat))
            feats[f"e_{name}_entropy"] = float(FeatureExtractor._shannon_entropy(flat))
            feats[f"e_{name}_energy"] = float(np.sum(flat ** 2))
        return feats

    # ------------------------------------------------------------------
    # Group F – PRNU-style Features
    # ------------------------------------------------------------------

    def _group_f_prnu(self, r: np.ndarray) -> Dict[str, float]:
        # Noise variance and energy
        noise_var = float(np.var(r))
        noise_energy = float(np.sum(r ** 2))

        # Residual correlation: how correlated is the residual with a
        # smoothed (low-pass) version of itself?
        from scipy.ndimage import uniform_filter
        smooth = uniform_filter(r, size=5)
        corr_val = float(np.corrcoef(r.ravel(), smooth.ravel())[0, 1])

        # Local noise consistency – std of per-block variances
        block_vars = self._local_block_variances(r)
        local_consistency = float(np.std(block_vars))
        local_mean_var = float(np.mean(block_vars))

        return {
            "f_noise_variance": noise_var,
            "f_noise_energy": noise_energy,
            "f_residual_correlation": corr_val,
            "f_local_consistency": local_consistency,
            "f_local_mean_var": local_mean_var,
        }

    def _local_block_variances(self, r: np.ndarray) -> np.ndarray:
        """Variance of each non-overlapping block of side *local_block_size*."""
        h, w = r.shape
        bs = self.local_block_size
        variances = []
        for i in range(0, h - bs + 1, bs):
            for j in range(0, w - bs + 1, bs):
                block = r[i : i + bs, j : j + bs]
                variances.append(float(np.var(block)))
        return np.array(variances) if variances else np.array([0.0])

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _shannon_entropy(values: np.ndarray) -> float:
        """Approximate Shannon entropy via histogram of 256 bins."""
        hist, _ = np.histogram(values, bins=256)
        hist = hist[hist > 0].astype(np.float64)
        prob = hist / hist.sum()
        return float(-np.sum(prob * np.log2(prob + EPS)))
