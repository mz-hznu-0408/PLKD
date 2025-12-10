# PLKD: Pattern Learning Knowledge Distillation for Single-Cell Annotation

## Package: `PLKD`

We created the python package called `PLKD` that uses `scanpy` ans `torch` to explainablely annotate cell type on single-cell RNA-seq data.

### Requirements

+ Linux/UNIX/Windows system
+ Python >= 3.8
+ torch == 1.7.1

### Create environment

```
conda create -n PLKD python=3.8
conda activate PLKD
conda install pytorch=1.7.1 torchvision=0.8.2 torchaudio=0.7.2 cudatoolkit=10.1 -c pytorch
pip install matplotlib==3.5.1
pip install scanpy==1.9.8
```

# Quick Start
We provide a ready-to-run end-to-end example in `test/tutorial.py`, covering Teacher training, Student distillation, and Student prediction.

```bash 
python test/tutorial.py
```

Script Functions:

- Automatically generate synthetic data with Celltype labels;

- Train PLKD (Teacher + Student), saving the output to `./PLKD_tutorial/`;

- Predict the query set using the latest Student weights and write the results to `test/demo_plkd_prediction.h5ad`;

- Provide the prediction accuracy and compute the latent representation of the UMAP, writing the coordinates back to AnnData and additionally outputting `test/demo_plkd_umap.pdf`.

Or you can use the real-data in `test/`, the separate single-cell atlases (demo datasets):
[https://1drv.ms/f/c/94a9f528230586fe/IgBnLK0axJkqR6ej9wQRq1XAAa5p3A4WS9wjXhKu8g29rsU?e=yADegF](https://1drv.ms/f/c/94a9f528230586fe/IgBnLK0axJkqR6ej9wQRq1XAAa5p3A4WS9wjXhKu8g29rsU?e=yADegF)

When running the [`tutorial.py`](test/tutorial.py), you can read the [`test/README.md`](test/README.md) for more details.
