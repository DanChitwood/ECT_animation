import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw
import sys
import h5py
import pickle
from io import BytesIO
import cv2
from ect import ECT, EmbeddedGraph

# --- Configuration Parameters ---
BOUND_RADIUS = 1
NUM_ECT_DIRECTIONS = 360
ECT_THRESHOLDS = np.linspace(0, BOUND_RADIUS, NUM_ECT_DIRECTIONS)
IMAGE_SIZE = (512, 512)

# --- Input/Output Paths (relative to the script location) ---
BASE_DIR = Path(__file__).parent.parent
SAVED_MODEL_SUB_DIR = BASE_DIR / "data" / "saved_leaf_model_data"
PCA_PARAMS_FILE = SAVED_MODEL_SUB_DIR / "leaf_pca_model_parameters.h5"
PCA_SCORES_LABELS_FILE = SAVED_MODEL_SUB_DIR / "original_pca_scores_and_geno_labels.h5"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "ect_plots"

# --- Coordinate Split Information ---
NUM_VEIN_COORDS = 1216
NUM_BLADE_COORDS = 456
TOTAL_COORDS = NUM_VEIN_COORDS + NUM_BLADE_COORDS
FLATTENED_COORD_DIM = TOTAL_COORDS * 2

# --- Helper Functions (copied from your original script) ---
def ect_coords_to_pixels(coords_ect: np.ndarray, image_size: tuple, bound_radius: float):
    if len(coords_ect) == 0:
        return np.array([])
    
    display_x_conceptual = coords_ect[:, 1]
    display_y_conceptual = coords_ect[:, 0]

    scale_factor = image_size[0] / (2 * bound_radius)
    offset_x = image_size[0] / 2
    offset_y = image_size[1] / 2 

    pixel_x = (display_x_conceptual * scale_factor + offset_x).astype(int)
    pixel_y = (-display_y_conceptual * scale_factor + offset_y).astype(int)
    
    return np.column_stack((pixel_x, pixel_y))

def create_ect_image_in_memory(ect_result, cmap_name: str = "gray"):
    if ect_result is None or ect_result.T.size == 0:
        return Image.new("RGB", IMAGE_SIZE, "black")
    
    fig, ax = plt.subplots(subplot_kw=dict(projection="polar"),
                           figsize=(IMAGE_SIZE[0]/100, IMAGE_SIZE[1]/100), dpi=100)
    thetas = ect_result.directions.thetas
    thresholds = ect_result.thresholds
    THETA, R = np.meshgrid(thetas, thresholds)
    
    # Ensure ect_result.T has the correct shape for pcolormesh
    if THETA.shape != ect_result.T.shape:
        print(f"Warning: Meshgrid shape {THETA.shape} does not match ECT data shape {ect_result.T.shape}. Attempting to reshape.")
        ect_data = ect_result.T.reshape(thresholds.shape[0], thetas.shape[0])
    else:
        ect_data = ect_result.T

    im = ax.pcolormesh(THETA, R, ect_data, cmap=cmap_name) 
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim([0, BOUND_RADIUS])
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buffer = BytesIO()
    plt.savefig(buffer, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buffer.seek(0)
    
    return Image.open(buffer).convert("RGB")

def create_combined_viz(ect_pil_image: Image, overlay_coords: np.ndarray,
                         overlay_color: tuple, overlay_alpha: float,
                         overlay_type: str = "points"):
    ect_img = ect_pil_image.convert("RGBA")
    img_width, img_height = ect_img.size

    composite_overlay = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw_composite = ImageDraw.Draw(composite_overlay)

    if overlay_coords is not None and overlay_coords.size > 0:
        pixel_coords = ect_coords_to_pixels(overlay_coords, IMAGE_SIZE, BOUND_RADIUS)
        fill_color_with_alpha = (overlay_color[0], overlay_color[1], overlay_color[2], int(255 * overlay_alpha))

        if overlay_type == "mask_pixels":
            if len(pixel_coords) >= 3:
                polygon_points = [(int(p[0]), int(p[1])) for p in pixel_coords]
                draw_composite.polygon(polygon_points, fill=fill_color_with_alpha)
            else:
                for x, y in pixel_coords:
                    if 0 <= x < img_width and 0 <= y < img_height:
                        draw_composite.point((x, y), fill=fill_color_with_alpha)
        elif overlay_type == "points":
            point_radius = 2
            for x, y in pixel_coords:
                if 0 <= x < img_width and 0 <= y < img_height:
                    draw_composite.ellipse([x - point_radius, y - point_radius,
                                             x + point_radius, y + point_radius],
                                            fill=fill_color_with_alpha)
    
    final_combined_img = Image.alpha_composite(ect_img, composite_overlay).convert("RGB")
    return final_combined_img

def load_pca_model_data(pca_params_file: Path, pca_scores_labels_file: Path):
    print("Loading PCA model data...")
    if not pca_params_file.exists() or not pca_scores_labels_file.exists():
        raise FileNotFoundError("PCA model files not found. Please check paths.")

    pca_data = {}
    try:
        with h5py.File(pca_params_file, 'r') as f:
            pca_data['components'] = f['components'][:]
            pca_data['mean'] = f['mean'][:]
            pca_data['explained_variance'] = f['explained_variance'][:]
            pca_data['n_components'] = f.attrs['n_components']
            
        with h5py.File(pca_scores_labels_file, 'r') as f:
            pca_data['original_pca_scores'] = f['pca_scores'][:]
            pca_data['original_geno_labels'] = np.array([s.decode('utf-8') for s in f['geno_labels'][:]])
            
        print("PCA model data loaded successfully.")
        return pca_data
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading PCA data: {e}")

def inverse_transform_pca(pca_scores: np.ndarray, pca_components: np.ndarray, pca_mean: np.ndarray):
    reconstructed_data = np.dot(pca_scores, pca_components) + pca_mean
    return reconstructed_data

def rotate_coords_2d(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    if coords.size == 0:
        return np.array([])
    
    angle_rad = np.deg2rad(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    
    rotated_coords = coords @ rot_matrix.T
    return rotated_coords

def find_robust_affine_transformation_matrix(src_points: np.ndarray, dst_points: np.ndarray):
    if len(src_points) < 3 or len(dst_points) < 3:
        if len(src_points) == 0:
            return np.eye(3)
        raise ValueError(f"Need at least 3 points to compute affine transformation. Got {len(src_points)}.")
    
    chosen_src_pts = []
    chosen_dst_pts = []
    
    indices = np.arange(len(src_points))
    num_attempts = min(len(src_points) * (len(src_points) - 1) * (len(src_points) - 2) // 6, 1000)

    for _ in range(num_attempts):
        if len(src_points) >= 3:
            selected_indices = np.random.choice(indices, 3, replace=False)
            p1_src, p2_src, p3_src = src_points[selected_indices]
            p1_dst, p2_dst, p3_dst = dst_points[selected_indices]
            
            area_val = (p1_src[0] - p3_src[0]) * (p2_src[1] - p1_src[1]) - \
                       (p1_src[0] - p2_src[0]) * (p3_src[1] - p1_src[1])
            
            if np.abs(area_val) > 1e-6:
                chosen_src_pts = np.float32([p1_src, p2_src, p3_src])
                chosen_dst_pts = np.float32([p1_dst, p2_dst, p3_dst])
                break
    
    if len(chosen_src_pts) < 3:
        raise ValueError("Could not find 3 non-collinear points for affine transformation. Shape is likely degenerate or a line.")

    M_2x3 = cv2.getAffineTransform(chosen_src_pts, chosen_dst_pts)
    
    if M_2x3.shape != (2, 3):
        raise ValueError(f"cv2.getAffineTransform returned a non-(2,3) matrix: {M_2x3.shape}")

    affine_matrix_3x3 = np.vstack([M_2x3, [0, 0, 1]])
    
    return affine_matrix_3x3

def apply_transformation_with_affine_matrix(points: np.ndarray, affine_matrix: np.ndarray):
    if points.size == 0:
        return np.array([])
        
    if points.ndim == 1:
        if points.shape[0] == 2:
            points = points.reshape(1, 2)
        else:
            raise ValueError(f"Input 'points' is 1D but not a single (x,y) pair. Got shape: {points.shape}")
        
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Input 'points' must be a (N, 2) array. Got shape: {points.shape}")

    if affine_matrix.shape != (3, 3):
        raise ValueError(f"Input 'affine_matrix' must be (3, 3). Got shape: {affine_matrix.shape}")

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    if points_homogeneous.shape[1] != affine_matrix.T.shape[0]:
        raise ValueError(f"matmul: Input operand 1 has a mismatch in its core dimension 0. Expected {points_homogeneous.shape[1]}, got {affine_matrix.T.shape[0]}.")

    transformed_homogeneous = points_homogeneous @ affine_matrix.T
    return transformed_homogeneous[:, :2]

# --- Main Script Logic ---
def plot_ect_for_random_leaf():
    print("--- Starting ECT Visualization for a Random Leaf ---")

    # 1. Setup Output Directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ensuring output directory exists: {PLOTS_DIR}")

    # 2. Load PCA Data
    try:
        pca_data = load_pca_model_data(PCA_PARAMS_FILE, PCA_SCORES_LABELS_FILE)
    except Exception as e:
        print(f"An unexpected error occurred while loading PCA data: {e}")
        print("Exiting due to PCA data loading error.")
        return

    # 3. Select a Random Leaf
    original_pca_scores = pca_data['original_pca_scores']
    num_samples = len(original_pca_scores)
    random_index = np.random.randint(0, num_samples)
    random_pca_score = original_pca_scores[random_index]
    random_label = pca_data['original_geno_labels'][random_index]
    print(f"\nSelected a random leaf (index: {random_index}, class: {random_label}).")

    # 4. Inverse Transform PCA to get coordinates
    flat_coords = inverse_transform_pca(
        random_pca_score.reshape(1, -1),
        pca_data['components'],
        pca_data['mean']
    ).flatten()
    
    if len(flat_coords) != FLATTENED_COORD_DIM:
        print(f"Error: Reconstructed coordinates have unexpected dimension {len(flat_coords)}. Expected {FLATTENED_COORD_DIM}.")
        return

    coords_2d = flat_coords.reshape(TOTAL_COORDS, 2)
    raw_vein_coords = coords_2d[:NUM_VEIN_COORDS]
    raw_blade_coords = coords_2d[NUM_VEIN_COORDS:]

    # Apply a fixed rotation for consistent orientation
    fixed_rotation_angle_deg = -90
    rotated_raw_blade_coords = rotate_coords_2d(raw_blade_coords, fixed_rotation_angle_deg)
    rotated_raw_vein_coords = rotate_coords_2d(raw_vein_coords, fixed_rotation_angle_deg)

    # 5. Initialize ECT Calculator
    ect_calculator = ECT(num_dirs=NUM_ECT_DIRECTIONS, thresholds=ECT_THRESHOLDS, bound_radius=BOUND_RADIUS)
    print("Initialized ECT calculator.")

    # 6. Process Blade Data
    print("\nProcessing blade...")
    ect_result_blade = None
    transformed_blade_pixels = np.array([])
    if len(np.unique(rotated_raw_blade_coords, axis=0)) >= 3:
        G_blade = EmbeddedGraph()
        G_blade.add_cycle(rotated_raw_blade_coords)
        original_G_blade_coord_matrix = G_blade.coord_matrix.copy()

        G_blade.center_coordinates(center_type="origin")
        G_blade.transform_coordinates()
        G_blade.scale_coordinates(BOUND_RADIUS)

        if G_blade.coord_matrix.shape[0] > 2 and not np.all(G_blade.coord_matrix == 0):
            ect_result_blade = ect_calculator.calculate(G_blade)
            ect_affine_matrix_blade = find_robust_affine_transformation_matrix(original_G_blade_coord_matrix, G_blade.coord_matrix)
            transformed_blade_pixels = apply_transformation_with_affine_matrix(rotated_raw_blade_coords, ect_affine_matrix_blade)
    else:
        print("Warning: Insufficient unique blade points. Skipping blade ECT.")

    # 7. Process Vein Data
    print("Processing vein...")
    ect_result_vein = None
    transformed_vein_pixels = np.array([])
    if len(rotated_raw_vein_coords) >= 1:
        G_vein = EmbeddedGraph()
        if len(rotated_raw_vein_coords) >= 2:
            G_vein.add_cycle(rotated_raw_vein_coords)
        else:
            G_vein.add_points(rotated_raw_vein_coords)
        
        original_G_vein_coord_matrix = G_vein.coord_matrix.copy()

        G_vein.center_coordinates(center_type="origin")
        G_vein.transform_coordinates()
        G_vein.scale_coordinates(BOUND_RADIUS)

        if G_vein.coord_matrix.shape[0] > 0 and not np.all(G_vein.coord_matrix == 0):
            ect_result_vein = ect_calculator.calculate(G_vein)
            ect_affine_matrix_vein = find_robust_affine_transformation_matrix(original_G_vein_coord_matrix, G_vein.coord_matrix)
            transformed_vein_pixels = apply_transformation_with_affine_matrix(rotated_raw_vein_coords, ect_affine_matrix_vein)
    else:
        print("Warning: Insufficient vein points. Skipping vein ECT.")

    # 8. Create and Save Visualizations
    print("Creating and saving plots...")

    # Blade ECT and Combined Viz
    ect_img_blade = create_ect_image_in_memory(ect_result_blade, cmap_name="inferno")
    ect_img_blade.save(PLOTS_DIR / "blade_ect.png")
    
    combined_blade_viz = create_combined_viz(ect_img_blade, transformed_blade_pixels, (255, 255, 255), 0.2, "mask_pixels")
    combined_blade_viz.save(PLOTS_DIR / "combined_blade_viz.png")

    # Vein ECT and Combined Viz
    ect_img_vein = create_ect_image_in_memory(ect_result_vein, cmap_name="inferno")
    ect_img_vein.save(PLOTS_DIR / "vein_ect.png")

    combined_vein_viz = create_combined_viz(ect_img_vein, transformed_vein_pixels, (255, 255, 255), 1.0, "mask_pixels")
    combined_vein_viz.save(PLOTS_DIR / "combined_vein_viz.png")

    print("\nVisualizations saved successfully to the 'ect_plots' directory.")
    print("--- Script Finished ---")


if __name__ == "__main__":
    plot_ect_for_random_leaf()