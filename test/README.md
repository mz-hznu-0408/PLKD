
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

1. Download the dataset files from the provided OneDrive link (optional)
2. Place the downloaded files in this `test/` directory
3. Run the tutorial script:
   ```bash
   python3 tutorial.py
   ```

## Automatic Data Generation

The `tutorial.py` script includes an automatic data generation feature:
- If `demo_train.h5ad` and `demo_test.h5ad` are missing, the script will create synthetic datasets
- If `demo_base_profiles.npy` is present, it will use these profiles for generating consistent synthetic data
- If `demo_base_profiles.npy` is missing, the script will create new base profiles

## Other Required Files

The script also uses built-in GMT (Gene Set Matrix) files that are already included in the PLKD package:
- Located in `PLKD/resources/` directory
- The tutorial uses `human_gobp` by default

## Note

While the tutorial can run with automatically generated synthetic data, using the provided real datasets will yield better demonstration results and more accurate performance evaluation.
