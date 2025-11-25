import PLKD
import scanpy as sc
import numpy as np
import warnings
import torch
import os

warnings.filterwarnings("ignore")

# Check GPU
print("Torch version:", torch.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_capability(device=None), torch.cuda.get_device_name(device=None))
else:
    print("GPU not available, using CPU")

# Load Data
# Ensure these files exist in the current directory or provide correct paths
if not os.path.exists('demo_train.h5ad') or not os.path.exists('demo_test.h5ad'):
    print("Please ensure 'demo_train.h5ad' and 'demo_test.h5ad' are in the current directory.")
    # You might want to download them or use your own data
    # For this tutorial script to work out-of-the-box, we assume they exist.
else:
    print("Loading data...")
    ref_adata = sc.read('demo_train.h5ad')
    ref_adata = ref_adata[:, ref_adata.var_names]
    print("Reference data:", ref_adata)
    print(ref_adata.obs.Celltype.value_counts())

    query_adata = sc.read('demo_test.h5ad')
    query_adata = query_adata[:, ref_adata.var_names]
    print("Query data:", query_adata)
    print(query_adata.obs.Celltype.value_counts())

    # Training
    print("\n--- Starting PLKD Training ---")
    # This will train a Teacher (PLKD Teacher) and then distill to a Student (MLP)
    PLKD.train_plkd(
        ref_adata,
        gmt_path='human_gobp',  # Using built-in mask
        label_name='Celltype',
        epochs=3,
        project='PLKD_demo',
        alpha=0.5,
        temperature=4.0,
        student_hidden=[256, 128]
    )
    print("Training finished. Output saved to ./PLKD_demo")

    # Prediction
    print("\n--- Starting Prediction with Student Model ---")
    # Load the trained Student model weights
    # Note: The weight file name depends on the number of epochs. 
    # Since we ran for 3 epochs (0, 1, 2), the last model is student_model-2.pth
    student_weight_path = './PLKD_demo/student_model-2.pth'

    if os.path.exists(student_weight_path):
        new_adata = PLKD.pre_student(
            query_adata,
            model_weight_path=student_weight_path,
            project='PLKD_demo',
            hidden_dims=[256, 128] # Must match training config
        )

        # Save results
        new_adata.write('demo_plkd_result.h5ad')
        print("Prediction finished. Results saved to 'demo_plkd_result.h5ad'")
        print(new_adata)
        
        # Simple Analysis/Visualization (if running in an environment with plotting support)
        try:
            print("\n--- Performing basic analysis on results ---")
            new_adata.raw = new_adata
            # Note: new_adata.X contains latent features, so we don't need standard preprocessing for X
            # We can directly compute neighbors and UMAP on latent features
            sc.pp.neighbors(new_adata, use_rep='X')
            sc.tl.umap(new_adata)
            print("UMAP computed. You can visualize 'demo_plkd_result.h5ad' in a notebook.")
        except Exception as e:
            print(f"Analysis step failed: {e}")

    else:
        print(f"Model file {student_weight_path} not found. Training might have failed.")

