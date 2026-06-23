"""
pipeline.py
-----------
Ties together preprocessing → denoising → residual extraction →
feature extraction into a single callable object used by both the
training script and the inference script.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from preprocessing import ImagePreprocessor
from denoising import WaveletDenoiser
from residual import ResidualExtractor
from features import FeatureExtractor


class ForensicPipeline:
    """End-to-end PRNU-inspired forensic feature pipeline.

    Parameters
    ----------
    target_size : tuple of int
        Resize target passed to :class:`ImagePreprocessor`.
    residual_output_dir : str, optional
        If given, residual images are saved here.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        residual_output_dir: Optional[str] = None,
    ) -> None:
        self.preprocessor = ImagePreprocessor(target_size=target_size)
        self.denoiser = WaveletDenoiser(wavelet="db8")
        self.residual_extractor = ResidualExtractor(
            output_dir=residual_output_dir
        )
        self.feature_extractor = FeatureExtractor()

    # ------------------------------------------------------------------

    def process_image(
        self,
        image_path: str,
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        """Full pipeline for a single image.

        Returns
        -------
        features : dict
        original : np.ndarray  (H, W) float64 [0,1]
        denoised : np.ndarray  (H, W) float64 [0,1]
        residual : np.ndarray  (H, W) float64  (normalised)
        """
        name = Path(image_path).name

        original = self.preprocessor.load(image_path)
        denoised = self.denoiser.denoise(original)
        residual = self.residual_extractor.extract(original, denoised, name)
        feature_dict = self.feature_extractor.extract(residual)

        return feature_dict, original, denoised, residual
