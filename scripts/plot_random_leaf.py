import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import h5py # Make sure this is installed!

# --- Configuration ---
# Adjust these paths relative to the scripts directory
# ECT_animation/scripts/
# Assumes:
# ECT_animation/
# ├── data/
# │   └── saved_leaf_model_data/
# │       └── ... (PCA files here)
# ├── scripts/
# │   └── plot_random_leaf.py
# └── outputs/
#     └── (will be created)

BASE_DIR = Path(__file__).parent.parent # ECT_animation directory
DATA_DIR = BASE_DIR / "data"
SAVED_MODEL_DIR = DATA_DIR / "saved_leaf_model_data"
OUTPUT_DIR = BASE_DIR / "outputs"

# Input PCA files
PCA_PARAMS_FILE = SAVED_MODEL_DIR / "leaf_pca_model_parameters.h5"
PCA_SCORES_LABELS_FILE = SAVED_MODEL_DIR / "original_pca_scores_and_geno_labels.h5"

# Output directory for plots
OUTPUT_PLOTS_DIR = OUTPUT_DIR / "random_leaf_plots"

# Coordinate structure from your original code
NUM_VEIN_COORDS = 1216
NUM_BLADE_COORDS = 456
TOTAL_COORDS = NUM_VEIN_COORDS + NUM_BLADE_COORDS # Should be 1672
FLATTENED_COORD_DIM = TOTAL_COORDS * 2 # 1672 * 2 = 3344

# Global random seed for reproducibility
GLOBAL_RANDOM_SEED = 42

# --- Helper Functions ---

def load_pca_model_data(pca_params_file: Path, pca_scores_labels_file: Path):
    """
    Loads PCA model parameters and original PCA scores/labels from .h5 files.
    """
    pca_data = {}
    try:
        with h5py.File(pca_params_file, 'r') as f:
            pca_data['components'] = f['components'][:]
            pca_data['mean'] = f['mean'][:]
            pca_data['explained_variance'] = f['explained_variance'][:]
            pca_data['n_components'] = f.attrs['n_components']
        print(f"Successfully loaded PCA parameters from: {pca_params_file}")

        with h5py.File(pca_scores_labels_file, 'r') as f:
            pca_data['original_pca_scores'] = f['pca_scores'][:]
            # Decode byte strings to UTF-8
            pca_data['original_geno_labels'] = np.array([s.decode('utf-8') for s in f['geno_labels'][:]])
        print(f"Successfully loaded PCA scores and labels from: {pca_scores_labels_file}")

    except FileNotFoundError as e:
        print(f"Error: PCA file not found. Please check the path: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing expected dataset or attribute in HDF5 file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading PCA data: {e}")
        sys.exit(1)
    return pca_data

def inverse_transform_pca(pca_scores: np.ndarray, pca_components: np.ndarray, pca_mean: np.ndarray):
    """
    Inverse transforms PCA scores back to the original flattened coordinate space.
    Assumes pca_components are (n_components, n_features) and pca_mean is (n_features,).
    """
    # Ensure pca_scores is 2D for dot product
    if pca_scores.ndim == 1:
        pca_scores = pca_scores.reshape(1, -1)

    # Perform the inverse transform: scores * components + mean
    reconstructed_data = np.dot(pca_scores, pca_components) + pca_mean
    return reconstructed_data.flatten() # Return as a 1D array

def plot_leaf_shape(flat_coords: np.ndarray, title: str, save_path: Path):
    """
    Plots the leaf shape from flattened coordinates, separating vein and blade.
    """
    if flat_coords.size == 0:
        print("No coordinates to plot.")
        return

    try:
        coords_2d = flat_coords.reshape(TOTAL_COORDS, 2)
        vein_coords = coords_2d[:NUM_VEIN_COORDS]
        blade_coords = coords_2d[NUM_VEIN_COORDS:]
    except ValueError as e:
        print(f"Error reshaping coordinates: {e}. Expected {FLATTENED_COORD_DIM} elements, but got {flat_coords.size}.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot Vein
    if vein_coords.shape[0] > 0:
        ax.plot(vein_coords[:, 0], vein_coords[:, 1], marker='.', color='blue', linestyle='-', markersize=2, label='Vein')

    # Plot Blade
    if blade_coords.shape[0] > 0:
        ax.plot(blade_coords[:, 0], blade_coords[:, 1], marker='.', color='green', linestyle='-', markersize=2, label='Blade')

    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_aspect('equal', adjustable='box') # Ensure aspect ratio is equal
    ax.grid(True)
    ax.legend()

    plt.savefig(save_path)
    print(f"Saved leaf plot to: {save_path}")
    plt.close(fig) # Close the figure to free up memory

# --- Main Execution ---

def main():
    print("--- Starting Random Leaf Plotting Script ---")

    # Ensure output directory exists
    OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {OUTPUT_PLOTS_DIR}")

    # Load PCA data
    pca_data = load_pca_model_data(PCA_PARAMS_FILE, PCA_SCORES_LABELS_FILE)

    # Set random seed for reproducibility of sample selection
    np.random.seed(GLOBAL_RANDOM_SEED)

    # Select a random PCA score
    num_original_samples = pca_data['original_pca_scores'].shape[0]
    random_index = np.random.randint(0, num_original_samples)
    random_pca_score = pca_data['original_pca_scores'][random_index]
    random_geno_label = pca_data['original_geno_labels'][random_index]

    print(f"\nSelected random sample at index: {random_index}")
    print(f"Associated Genotype Label: {random_geno_label}")

    # Inverse transform the random PCA score to get flattened coordinates
    flat_coords = inverse_transform_pca(
        random_pca_score,
        pca_data['components'],
        pca_data['mean']
    )

    # Check if the reconstructed flattened coordinates have the expected dimension
    if len(flat_coords) != FLATTENED_COORD_DIM:
        print(f"Error: Reconstructed flattened coordinates have unexpected dimension {len(flat_coords)}. Expected {FLATTENED_COORD_DIM}. Cannot plot.")
        sys.exit(1)

    # Plot the leaf shape
    plot_title = f"Random Leaf Sample (Genotype: {random_geno_label}, Index: {random_index})"
    plot_filename = f"random_leaf_{random_index:05d}.png"
    plot_path = OUTPUT_PLOTS_DIR / plot_filename

    plot_leaf_shape(flat_coords, plot_title, plot_path)

    print("\n--- Random Leaf Plotting Script Finished ---")

if __name__ == "__main__":
    main()