"""
preprocessing.py
----------------
Handles image loading, validation, grayscale conversion,
resizing, and pixel normalization before forensic analysis.

Why preprocessing matters for PRNU forensics:
- Grayscale removes colour channel noise, isolating sensor-pattern noise.
- Fixed resize ensures all feature vectors have identical dimensionality.
- Normalisation stabilises numerical ranges across images from different cameras/models.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError


# ──────────────────────────────────────────
# Constants
# ──────────────────────────────────────────
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_SIZE: Tuple[int, int] = (512, 512)


# ──────────────────────────────────────────
# ImagePreprocessor
# ──────────────────────────────────────────
class ImagePreprocessor:
    """Loads and prepares an image for forensic analysis.

    Parameters
    ----------
    target_size : tuple of int, optional
        (width, height) to which every image is resized.
        Default is (512, 512).
    """

    def __init__(self, target_size: Tuple[int, int] = DEFAULT_SIZE) -> None:
        self.target_size = target_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, image_path: str) -> np.ndarray:
        """Load, validate, and preprocess a single image.

        Parameters
        ----------
        image_path : str
            Absolute or relative path to the image file.

        Returns
        -------
        np.ndarray
            Float64 array of shape (H, W) with values in [0, 1].

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file extension is unsupported or the image cannot be decoded.
        """
        path = Path(image_path)

        # ── Existence check ──
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # ── Extension check ──
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported extension '{path.suffix}'. "
                f"Accepted: {SUPPORTED_EXTENSIONS}"
            )

        # ── Decode ──
        try:
            pil_img = Image.open(path).convert("L")  # grayscale
        except UnidentifiedImageError as exc:
            raise ValueError(f"Cannot decode image '{image_path}': {exc}") from exc

        # ── Resize ──
        pil_img = pil_img.resize(self.target_size, Image.LANCZOS)

        # ── Convert & normalise ──
        arr = np.array(pil_img, dtype=np.float64) / 255.0
        return arr

    def load_batch(
        self,
        folder: str,
        label: int,
        max_images: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load all supported images from a directory.

        Parameters
        ----------
        folder : str
            Directory containing image files.
        label : int
            Class label (0 = real, 1 = AI-generated).
        max_images : int, optional
            Cap on the number of images to load per folder.

        Returns
        -------
        images : np.ndarray  shape (N, H, W)
        labels : np.ndarray  shape (N,)
        """
        folder_path = Path(folder)
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {folder}")

        paths = sorted(
            p for p in folder_path.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if max_images is not None:
            paths = paths[:max_images]

        images, labels = [], []
        for p in paths:
            try:
                img = self.load(str(p))
                images.append(img)
                labels.append(label)
            except (ValueError, FileNotFoundError) as exc:
                print(f"[WARN] Skipping {p.name}: {exc}")

        if not images:
            raise RuntimeError(f"No valid images found in '{folder}'.")

        return np.array(images, dtype=np.float64), np.array(labels, dtype=np.int32)
