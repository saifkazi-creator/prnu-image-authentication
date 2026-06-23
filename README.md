# PRNU-Inspired AI Image Forensic Detector

Classifies images as **Real Camera** or **AI Generated** using sensor-noise
residual analysis and machine learning — no deep learning required.

```
Image
 → DB8 Wavelet Denoising
 → Noise Residual (Original − Denoised)
 → 42-dimensional Forensic Feature Vector
 → Classifier (RF / XGB / SVM / LGBM / MLP)
 → Real / AI  +  Confidence Score
```

---

## Project Structure

```
ai_image_detector/
├── dataset/
│   ├── real/          ← real camera photographs
│   └── ai/            ← AI-generated images
├── src/
│   ├── preprocessing.py   grayscale, resize, normalise
│   ├── denoising.py       DB8 multi-level wavelet denoising
│   ├── residual.py        noise residual extraction
│   ├── features.py        6-group forensic feature extraction
│   ├── pipeline.py        end-to-end pipeline
│   ├── train.py           training script
│   ├── evaluate.py        metrics + all visualisations
│   └── predict.py         single-image inference CLI
├── models/                saved model.pkl / scaler.pkl
├── outputs/
│   ├── features/          features.csv
│   ├── residuals/         noise residual images
│   └── figures/           all EDA + evaluation plots
├── streamlit_app.py       interactive web UI
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

Populate the two sub-directories:

```
dataset/real/   ← jpg / jpeg / png  (real photos from camera)
dataset/ai/     ← jpg / jpeg / png  (AI-generated images)
```

At least **100 images per class** is recommended for meaningful training;
**500+** per class gives robust results.

---

## Training

```bash
# Full training run
python src/train.py --dataset dataset/ --output outputs/

# Limit images per class for a quick experiment
python src/train.py --dataset dataset/ --output outputs/ --max-imgs 200

# Force feature re-extraction (ignores existing features.csv)
python src/train.py --dataset dataset/ --output outputs/ --rebuild-csv
```

What happens:
1. Every image is passed through the forensic pipeline → 42 features.
2. Features are saved to `outputs/features/features.csv`.
3. EDA figures are saved to `outputs/figures/`.
4. Five classifiers are trained with 5-fold cross-validation.
5. The best model by F1 is saved to `outputs/models/model.pkl`.
6. Feature importance and SHAP plots are saved.

---

## Inference (CLI)

```bash
python src/predict.py --image path/to/image.jpg
```

Sample output:

```
──────────────────────────────────────────────────
  Prediction  :  Real Image
  Confidence  :  94.37 %
──────────────────────────────────────────────────

  Top forensic features:
    f_noise_energy                    :   8234.217300
    b_total_energy                    :   8234.217300
    c_fft_mean_mag                    :    432.881200
    ...
```

---

## Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

Features:
- Upload any image (JPG / JPEG / PNG)
- View original, denoised, and residual side by side
- Prediction verdict + confidence gauge
- Interactive bar chart of all 42 features
- Feature-group energy radar chart
- Full feature table
- Forensic interpretation guide

---

## Feature Groups Explained

### Why these features distinguish Real from AI

| Group | Features | Rationale |
|---|---|---|
| **A – Statistics** | mean, variance, std, RMS, skewness, kurtosis, entropy | Real camera PRNU is signal-dependent and slightly non-Gaussian. AI residuals are often more Gaussian (diffusion) or heavy-tailed (GAN). |
| **B – Energy** | total_energy, avg_energy | Genuine sensor noise injects a consistent, predictable amount of PRNU energy. AI models produce either very low or irregularly patterned energy. |
| **C – FFT** | mean/var magnitude, spectral entropy, peak mag, high/low freq ratios | GANs leave periodic spectral peaks from transposed-convolution strides. Diffusion U-Nets inflate low-frequency residual energy. Flat spectrum → real camera. |
| **D – Autocorrelation** | max, mean, variance of off-peak autocorrelation | Sensor noise is spatially nearly white → fast autocorrelation decay. Convolution layers in AI generators produce long-range spatial structure. |
| **E – Wavelet sub-bands** | LL / LH / HL / HH mean, variance, entropy, energy | Real cameras concentrate noise in the HH (finest detail) sub-band. AI generators with coarse-to-fine synthesis leak energy into LL. |
| **F – PRNU-style** | noise variance, energy, residual correlation, local consistency | Classical PRNU fingerprinting metrics adapted for binary classification. AI-generated residuals are spatially inconsistent in characteristic ways. |

---

## Recommended Datasets

### Real Camera Images
| Dataset | Source | Notes |
|---|---|---|
| **RAISE** (8,156 raw photos) | https://loki.disi.unitn.it/RAISE/ | Diverse cameras, unprocessed RAW |
| **Dresden Image Database** | https://forensics.inf.tu-dresden.de/ddimgdb/ | Multi-device, standard for PRNU research |
| **MIT-Adobe FiveK** | https://data.csail.mit.edu/graphics/fivek/ | 5,000 DSLR shots |
| **COCO** (real photos) | https://cocodataset.org/ | Large-scale, diverse scenes |

### AI-Generated Images
| Dataset / Source | Generator | Notes |
|---|---|---|
| **CIFAKE** (60,000 AI images) | Stable Diffusion v1.4 | Paired with CIFAR-10 real images |
| **DiffusionDB** | Stable Diffusion | 14 M images with prompts |
| **JourneyDB** | Midjourney v5 | 4 M high-quality images |
| **GenImage** benchmark | SD, DALL-E, Midjourney, Wukong | Multi-generator benchmark dataset |
| **DALL-E 3 API** | DALL-E 3 | Generate via OpenAI API |
| **Flux.1 inference** | Flux (Black Forest Labs) | Run locally via diffusers |

### Quick-start mix suggestion
For a balanced binary classification experiment, combine:
- **Real**: ~1,000 images from RAISE or Dresden
- **AI**: ~1,000 images from CIFAKE or DiffusionDB

---

## Reproducibility

All random seeds are set via `--seed` (default `42`).
The feature CSV is deterministic; re-running with the same seed yields identical splits.

---

## License

MIT — use freely with attribution.
