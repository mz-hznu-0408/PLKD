import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import PLKD

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
TRAIN_PATH = HERE / "demo_train.h5ad"
TEST_PATH = HERE / "demo_test.h5ad"
BASE_PROFILE_PATH = HERE / "demo_base_profiles.npy"
PROJECT = "PLKD_tutorial"
EPOCHS = 6
STUDENT_HIDDEN = [256, 128]
UMAP_FIG = HERE / "demo_plkd_umap.pdf"


def log_device():
    print(f"Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU not available, fallback to CPU")


def generate_synthetic_adata(
    path: Path,
    n_cells=200,
    n_genes=500,
    n_types=3,
    seed=0,
    base_profiles=None,
    save_profiles_path: Path = None,
):
    """
    Create a toy AnnData object with clear cell-type signals and save to disk.
    """
    rng = np.random.default_rng(seed)

    gene_names = [f"Gene{i}" for i in range(n_genes)]
    cell_ids = [f"Cell{i}" for i in range(n_cells)]

    # Create base profiles for each cell type to ensure the model can learn structure.
    created_profiles = False
    if base_profiles is None:
        base_profiles = rng.gamma(shape=2.0, scale=1.0, size=(n_types, n_genes))
        created_profiles = True
    else:
        base_profiles = np.asarray(base_profiles)
        if base_profiles.shape != (n_types, n_genes):
            raise ValueError("base_profiles shape mismatch with n_types/n_genes.")

    base_count = n_cells // n_types
    remainder = n_cells % n_types
    labels = []
    for t in range(n_types):
        count = base_count + (1 if t < remainder else 0)
        labels.extend([t] * count)
    labels = np.array(labels, dtype=np.int64)
    rng.shuffle(labels)

    expr = np.zeros((n_cells, n_genes), dtype=np.float32)

    for idx, label in enumerate(labels):
        profile = base_profiles[label] + rng.normal(scale=0.3, size=n_genes)
        expr[idx] = np.clip(profile, a_min=0, a_max=None)

    obs = pd.DataFrame({"Celltype": [f"Type{l}" for l in labels]}, index=cell_ids)
    var = pd.DataFrame(index=gene_names)

    adata = sc.AnnData(expr, obs=obs, var=var)
    adata.write(path)
    print(f"Synthetic dataset saved to {path}")

    if created_profiles and save_profiles_path is not None:
        np.save(save_profiles_path, base_profiles)

    return base_profiles


def ensure_datasets():
    """
    Ensure demo_train/demo_test files exist. If missing, auto-generate.
    """
    base_profiles = None
    if BASE_PROFILE_PATH.exists():
        base_profiles = np.load(BASE_PROFILE_PATH)

    if not TRAIN_PATH.exists():
        print("demo_train.h5ad not found. Generating synthetic training data...")
        base_profiles = generate_synthetic_adata(
            TRAIN_PATH,
            seed=0,
            save_profiles_path=BASE_PROFILE_PATH,
        )
    if not TEST_PATH.exists():
        print("demo_test.h5ad not found. Generating synthetic testing data...")
        if base_profiles is None and BASE_PROFILE_PATH.exists():
            base_profiles = np.load(BASE_PROFILE_PATH)
        generate_synthetic_adata(TEST_PATH, seed=1, base_profiles=base_profiles)


def train_and_eval():
    log_device()
    ensure_datasets()

    print("\nLoading AnnData files...")
    ref_adata = sc.read(TRAIN_PATH)
    query_adata = sc.read(TEST_PATH)
    query_adata = query_adata[:, ref_adata.var_names]

    print(ref_adata)
    print(ref_adata.obs["Celltype"].value_counts())
    print(query_adata)
    print(query_adata.obs["Celltype"].value_counts())

    print("\n--- Training PLKD (Teacher + Student) ---")
    PLKD.train_plkd(
        ref_adata,
        gmt_path="human_gobp",
        label_name="Celltype",
        epochs=EPOCHS,
        batch_size=16,
        project=PROJECT,
        alpha=0.5,
        temperature=4.0,
        student_hidden=STUDENT_HIDDEN,
        divergence_weight=1.0,
        self_entropy_weight=1.0,
    )
    print(f"Training finished. Outputs located in ./{PROJECT}")

    student_weight_path = Path(f"./{PROJECT}/student_model-{EPOCHS-1}.pth")
    if not student_weight_path.exists():
        raise FileNotFoundError(f"Student weights {student_weight_path} not found.")

    print("\n--- Running Student inference ---")
    pred_adata = PLKD.pre_student(
        query_adata,
        model_weight_path=str(student_weight_path),
        project=PROJECT,
        hidden_dims=STUDENT_HIDDEN,
    )

    pred_file = HERE / "demo_plkd_prediction.h5ad"
    pred_adata.write(pred_file)
    print(f"Inference finished. Prediction AnnData saved to {pred_file}")

    if "Celltype" in pred_adata.obs:
        preds = pred_adata.obs["Prediction"].astype("string")
        labels = pred_adata.obs["Celltype"].astype("string")
        accuracy = (preds == labels).mean()
        print(f"Student accuracy on synthetic test set: {accuracy:.3f}")
    else:
        print("Ground-truth labels missing in prediction AnnData; skipping accuracy.")

    try:
        print("Computing neighbors + UMAP on latent representation...")
        sc.pp.neighbors(pred_adata, use_rep="X")
        sc.tl.umap(pred_adata)
        color_keys = ["Prediction"]
        if "Celltype" in pred_adata.obs:
            color_keys.append("Celltype")
        sc.pl.umap(pred_adata, color=color_keys, ncols=len(color_keys), show=False)
        plt.savefig(UMAP_FIG, format="pdf", bbox_inches="tight")
        plt.close("all")
        pred_adata.write(pred_file)
        print(f"UMAP coordinates stored in AnnData and figure saved to {UMAP_FIG}")
    except Exception as exc:
        print(f"UMAP computation skipped: {exc}")


if __name__ == "__main__":
    train_and_eval()

