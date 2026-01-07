import cv2
import numpy as np
import sys
import os
import pywt 

prnu1 = np.load("prnu/prnu_cam1.npy")
prnu2 = np.load("prnu/prnu_cam2.npy")
T_low  = np.load("prnu/T_low.npy")
T_high = np.load("prnu/T_high.npy")

def center_crop(img, size=2048):
    h, w = img.shape
    size = min(size, h, w)
    half = size // 2
    return img[h//2-half:h//2+half, w//2-half:w//2+half]

def extract_residual_wavelet(img):
    img = center_crop(img)

    # Wavelet decomposition
    coeffs = pywt.wavedec2(img, wavelet='db8', level=1)

    # Remove approximation (scene)
    coeffs[0] = np.zeros_like(coeffs[0])

    # Reconstruct noise
    r = pywt.waverec2(coeffs, wavelet='db8')

    return r - np.mean(r)

def correlate(img, prnu):
    r = extract_residual_wavelet(img)
    h, w = r.shape
    p = prnu[:h, :w] - np.mean(prnu[:h, :w])
    return np.sum(r*p) / (np.linalg.norm(r)*np.linalg.norm(p) + 1e-8)

def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("❌ Could not read image")
        return

    img = img.astype(np.float32)

    c1 = correlate(img, prnu1)
    c2 = correlate(img, prnu2)
    corr = max(c1, c2)

    print(f"\nPRNU correlation: {corr:.6f}")

# === FINAL DECISION (threshold-based, authoritative) ===
    if corr >= T_high:
        decision = "REAL IMAGE (Camera authenticated)"
    elif corr < T_low:
        decision = "NOT AUTHENTIC (AI or unknown device)"
    else:
        decision = "UNCERTAIN"

    print(f"DECISION: {decision}")

# === EXPLANATION (interpretive, safe) ===
    if corr < 0:
        print(
        "Reason: Correlation is negative, indicating strong mismatch with sensor PRNU. "
        "High likelihood of AI-generated or synthetic origin."
        )
    elif corr < T_low:
        print(
        "Reason: Weak positive correlation without a camera match. "
        "Image could be AI-generated or from an unknown device."
        )
    elif corr < T_high:
        print(
        "Reason: Moderate correlation but insufficient for authentication. "
        "Image may be edited or from a different camera."
        )
    else:
        print(
        "Reason: Strong correlation with enrolled camera PRNU."
        )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    predict_image(sys.argv[1])
