# PRNU-Inspired AI Image Forensic Detector

Classifies images as **Real Camera** or **AI Generated** using sensor-noise residual analysis and machine learning — no deep learning required.

```
Image
 → DB8 Wavelet Denoising
 → Noise Residual (Original − Denoised)
 → 39-dimensional Forensic Feature Vector
 → Classifier (RF / XGB / SVM / LGBM / MLP)
 → Real / AI  +  Confidence Score
```

> **Trained on ~20,000 images** — 10,000 real camera photos (COCO 2017 + Kaggle real images) + 10,000 AI-generated images (DiffusionDB Stable Diffusion + Kaggle multi-generator dataset covering GAN, SDXL, and other generators).

---

## Results

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| RandomForest | 87.8% | 0.875 | 0.943 |
| XGBoost | 87.5% | 0.873 | 0.948 |
| SVM | 87.6% | 0.874 | 0.944 |
| LightGBM | 87.8% | 0.876 | 0.948 |
| **MLP** ✓ best | **88.2%** | **0.881** | **0.951** |

Evaluated on a held-out test set (stratified 80/20 split) across 20,000 images from multiple AI generators.

> Accuracy is lower than single-generator benchmarks by design — the model is trained on diverse AI generators (SD, GAN, SDXL) making it a harder and more realistic task. A model scoring 93% on SD-only data will fail on DALL·E or Midjourney images; this model generalises across generators.

---

## Project Structure

```
ai_image_detector/
│
├── dataset/                        ← images go here (not committed to git)
│   ├── real/                       ← real camera photographs (JPG/PNG)
│   └── ai/                         ← AI-generated images (JPG/PNG)
│
├── src/                            ← all source code
│   ├── preprocessing.py            grayscale, resize to 512×512, normalise
│   ├── denoising.py                DB8 multi-level wavelet denoising
│   ├── residual.py                 noise residual = original − denoised
│   ├── features.py                 39 forensic features across 6 groups
│   ├── pipeline.py                 end-to-end orchestrator
│   ├── train.py                    training script (5 classifiers)
│   ├── evaluate.py                 metrics, plots, SHAP analysis
│   └── predict.py                  single-image inference CLI
│
├── models/                         ← pre-trained model files
│   ├── model.pkl                   best trained classifier (MLP)
│   ├── scaler.pkl                  StandardScaler fitted on training data
│   └── feature_cols.pkl            feature column order for inference
│
├── outputs/                        ← auto-generated on training run
│   ├── features/
│   │   └── features.csv            extracted feature matrix (~20,000 rows × 39 features)
│   ├── residuals/                  noise residual PNG visualisations
│   └── figures/                    EDA plots, confusion matrices, ROC curves, SHAP
│
├── streamlit_app.py                interactive web UI
├── requirements.txt                pip dependencies
├── .gitignore
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd ai_image_detector

# 2. Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start — Inference Only (no training needed)

Pre-trained model files are included in the `models/` folder. Just install dependencies and run:

```bash
# CLI prediction
python src/predict.py --image path/to/your/image.jpg

# Web UI
streamlit run streamlit_app.py
```

---

## Dataset Setup (for training from scratch)

Populate the two sub-directories with images:

```
dataset/real/   ← JPG / PNG real camera photos  (min 256×256 px)
dataset/ai/     ← JPG / PNG AI-generated images (min 256×256 px)
```

**Recommended sources:**

| Class | Dataset | Link |
|---|---|---|
| Real | COCO 2017 val (5,000 images) | https://cocodataset.org/#download |
| Real | RAISE-8k (8,156 DSLR photos) | https://loki.disi.unitn.it/RAISE/ |
| AI | DiffusionDB 2m_first_5k | https://huggingface.co/datasets/poloclub/diffusiondb |
| AI | Kaggle AI vs Human dataset | https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset |
| AI | MS COCOAI (SD3, SDXL, DALL·E 3, MJ v6) | https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset |
| AI | DALL·E 3 1M dataset | https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions |

> **Important:** Use images that are at least **256×256 pixels**. Small thumbnails (e.g. CIFAKE at 32×32) will not work — the pipeline upscales them which destroys the noise residual.

> **Important:** Convert all WEBP images to JPG before training:
> ```bash
> python -c "
> from PIL import Image
> from pathlib import Path
> for f in Path('dataset/ai').rglob('*.webp'):
>     Image.open(f).convert('RGB').save(f.with_suffix('.jpg'), quality=95)
>     f.unlink()
> "
> ```

---

## Training

```bash
# Full training run
python src/train.py --dataset dataset/ --output outputs/

# Limit images per class (quick experiment)
python src/train.py --dataset dataset/ --output outputs/ --max-imgs 500

# Force re-extraction even if features.csv exists
python src/train.py --dataset dataset/ --output outputs/ --rebuild-csv
```

**What the training script does:**
1. Loads every image from `dataset/real/` and `dataset/ai/`
2. Runs each through the forensic pipeline → 39 features per image
3. Saves all features to `outputs/features/features.csv`
4. Trains 5 classifiers with 5-fold stratified cross-validation
5. Evaluates on a held-out 20% test set
6. Saves the best model to `models/model.pkl`
7. Generates evaluation plots, SHAP analysis, and feature importance charts

---

## Inference

**CLI:**
```bash
python src/predict.py --image path/to/image.jpg
```

Sample output:
```
──────────────────────────────────────────────────
  Prediction  :  AI Generated Image
  Confidence  :  88.74 %
──────────────────────────────────────────────────

  Top forensic features:
    f_noise_energy                    :   1823.441200
    b_total_energy                    :   1823.441200
    c_fft_spectral_entropy            :      6.982100
    d_acorr_max                       :      0.412300
    ...
```

**Streamlit UI:**
```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`. Features:
- Upload any JPG / PNG image
- View original, denoised, and noise residual side by side
- Prediction verdict with confidence gauge
- Interactive bar chart of all 39 forensic features
- Feature group energy radar chart
- Full feature value table
- Forensic interpretation guide

---

## How It Works

### The Core Idea

A real camera sensor has microscopic physical imperfections — dust, uneven pixel sensitivity, manufacturing variation. These imprint a unique invisible pattern onto every photo called **PRNU (Photo Response Non-Uniformity)**. It is too faint to see but survives in the noise residual.

AI generators have no physical sensor. Stable Diffusion, DALL·E, Midjourney etc. synthesise pixels mathematically from a neural network. Their residuals instead reveal generator artifacts — periodic patterns from GAN convolutional layers, spectral signatures from diffusion U-Net denoising steps.

### Pipeline

```
Step 1 — Preprocessing
  Convert to grayscale → resize to 512×512 → normalise to [0, 1]
  File: src/preprocessing.py

Step 2 — DB8 Wavelet Denoising
  Decompose with Daubechies-8 wavelet (5 levels)
  Threshold detail sub-bands with soft thresholding (MAD noise estimate)
  Reconstruct → denoised image (scene without noise)
  File: src/denoising.py

Step 3 — Noise Residual Extraction
  Residual = Original − Denoised
  Z-score normalise → zero mean, unit variance
  File: src/residual.py

Step 4 — Feature Extraction (39 features across 6 groups)
  File: src/features.py

Step 5 — Classification
  StandardScaler → trained MLP / SVM / RF / XGB / LGBM
  File: src/train.py
```

### Feature Groups

| Group | Features | Why it works |
|---|---|---|
| **A – Statistics** | mean, variance, std, RMS, skewness, kurtosis, entropy | Real PRNU is signal-dependent and non-Gaussian. AI residuals are more Gaussian (diffusion) or heavy-tailed (GAN). |
| **B – Energy** | total_energy, avg_energy | Sensors inject consistent PRNU energy. AI generators produce irregular energy levels. |
| **C – FFT** | mean/var magnitude, spectral entropy, peak, high/low freq ratio | GANs leave periodic spectral peaks from stride patterns. Diffusion U-Nets inflate low-frequency energy. |
| **D – Autocorrelation** | max, mean, variance | Real sensor noise decays in 1–2 pixels. AI convolutional layers create long-range spatial correlation. |
| **E – Wavelet sub-bands** | LL/LH/HL/HH mean, variance, entropy, energy | Real cameras concentrate noise in HH. AI generators leak energy into LL from coarse-to-fine synthesis. |
| **F – PRNU-style** | noise variance, energy, residual correlation, local consistency | True PRNU is spatially uniform. AI patterns vary locally in characteristic ways. |

---

## Known Limitations

- **Unseen generators:** Model accuracy drops on AI generators not in training data (e.g. Gemini, Flux). Adding images from new generators and retraining resolves this.
- **Resolution dependent:** Images below 256×256 px produce unreliable residuals. Do not use thumbnail datasets.
- **Heavy post-processing:** Images that have been heavily compressed, filtered, or resized after generation may fool the detector.
- **Not a deepfake detector:** This system is designed for fully AI-generated images, not face-swaps or partial manipulations.

---

## Future Improvements

- Add training data from Gemini (Imagen 3), DALL·E 3, Flux, and Midjourney
- Experiment with deep learning feature extractors (CNN-based residual analysis)
- Add batch inference support for processing entire folders
- Improve Streamlit UI with side-by-side comparison mode

---

## Reproducibility

All random seeds are set via `--seed` (default `42`). The feature CSV is deterministic — re-running with the same seed and dataset produces identical results.

---

## Requirements

```
numpy, scipy, Pillow, PyWavelets, scikit-learn
xgboost, lightgbm, shap
matplotlib, seaborn, pandas, joblib
streamlit, plotly, tqdm, scikit-image
```

Install all with: `pip install -r requirements.txt`

---

## License

MIT — use freely with attribution.
