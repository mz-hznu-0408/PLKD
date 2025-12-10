
# PLKD Tutorial

This directory contains a tutorial for the PLKD (Projection Learning Knowledge Distillation) framework.

## Dataset Files

When running the `tutorial.py` script, it uses the following files:

### Required Files (Optional)
These files are used for training and testing the PLKD model:
- `demo_train.h5ad` - Training dataset
- `demo_test.h5ad` - Testing dataset
- `demo_base_profiles.npy` - Base profiles for generating synthetic data (optional)

## Download Instructions

Due to GitHub file size limitations, these dataset files are hosted on OneDrive:

**Download URL:** https://1drv.ms/f/c/94a9f528230586fe/IgBnLK0axJkqR6ej9wQRq1XAAa5p3A4WS9wjXhKu8g29rsU?e=yADegF

## Usage

1. Download the dataset files from the provided OneDrive link
2. Place the downloaded files in this `test/` directory
3. Run the tutorial script:
   ```bash
   python tutorial.py
   ```

## Note

While the tutorial can run with automatically generated synthetic data, using the provided real datasets will yield better demonstration results and more accurate performance evaluation.
