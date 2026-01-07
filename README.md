# PRNU Image Authentication

Wavelet-domain PRNU based image authentication system for verifying whether an image originates from an enrolled camera sensor.

## Overview
This project implements a forensic image authentication pipeline using
Photo Response Non-Uniformity (PRNU) extracted in the wavelet domain.
The system verifies whether a test image matches an enrolled camera’s sensor fingerprint.

⚠️ This project focuses on **camera authentication**, not absolute AI image detection.

## Methodology
- Wavelet-domain noise residual extraction (DB8)
- PRNU fingerprint construction via averaging
- Correlation-based matching
- Threshold-based decision (REAL / NOT AUTHENTIC / UNCERTAIN)
