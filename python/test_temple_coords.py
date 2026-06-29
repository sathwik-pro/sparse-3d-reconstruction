import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper
import argparse


def run_sparse_reconstruction(left_img_path, right_img_path, correspondences_path, query_coords_path, camera_params_path):
    """
    Full pipeline for sparse 3D reconstruction from two images:
      1. Load images and known point correspondences
      2. Compute the Fundamental matrix (eight-point algorithm)
      3. Find epipolar correspondences for query points
      4. Compute Essential matrix from F and camera intrinsics
      5. Recover camera projection matrices from Essential matrix
      6. Triangulate 3D points and select the valid camera configuration
      7. Compute reprojection error
      8. Visualize and save the 3D point cloud
    """

    # -------------------------------------------------------------------------
    # Step 1: Load images and known point correspondences
    # -------------------------------------------------------------------------
    left_img  = plt.imread(left_img_path)
    right_img = plt.imread(right_img_path)

    # correspondences_path points to an .npz file with pre-labeled matching points
    corresp_file     = np.load(correspondences_path)
    known_pts_left   = corresp_file['pts1']   # Nx2 points in image 1
    known_pts_right  = corresp_file['pts2']   # Nx2 corresponding points in image 2

    # -------------------------------------------------------------------------
    # Step 2: Compute the Fundamental Matrix using the Eight-Point Algorithm
    # -------------------------------------------------------------------------
    # Normalization scalar = max image dimension, used to scale point coordinates
    img_scale = max(left_img.shape[0], left_img.shape[1])
    fund_mat  = sub.eight_point(known_pts_left, known_pts_right, img_scale)
    print("Fundamental Matrix F:")
    print(fund_mat)

    # -------------------------------------------------------------------------
    # Step 3: Load query points from image 1 and find their correspondences in image 2
    # -------------------------------------------------------------------------
    query_file = np.load(query_coords_path)

    # Handle different possible key layouts in the .npz file
    if 'pts1' in query_file:
        # Standard format: single array of Nx2 coordinates
        query_pts_left = query_file['pts1']
    elif 'x1' in query_file and 'y1' in query_file:
        # Alternate format: x and y coordinates stored separately
        query_pts_left = np.hstack((query_file['x1'], query_file['y1']))
    else:
        # Fallback: just take the first array in the file
        query_pts_left = query_file[query_file.files[0]]

    query_pts_left = query_pts_left.astype(float)

    print("\nFinding epipolar correspondences for the target coordinates...")

    # For each query point in image 1, find the best matching point in image 2
    # by searching along its epipolar line using Gaussian-weighted patch matching
    query_pts_right = sub.epipolar_correspondences(left_img, right_img, fund_mat, query_pts_left)

    # -------------------------------------------------------------------------
    # Step 4: Load camera intrinsics and compute the Essential Matrix
    # -------------------------------------------------------------------------
    cam_params_file  = np.load(camera_params_path)
    intrinsics_left  = cam_params_file['K1']   # 3x3 intrinsic matrix for camera 1
    intrinsics_right = cam_params_file['K2']   # 3x3 intrinsic matrix for camera 2

    # Essential matrix E = K2^T * F * K1
    # Encodes rotation and translation in calibrated (metric) coordinates
    ess_mat = sub.essential_matrix(fund_mat, intrinsics_left, intrinsics_right)
    print("\nEssential Matrix E:")
    print(ess_mat)

    # -------------------------------------------------------------------------
    # Step 5: Set up camera projection matrices
    # -------------------------------------------------------------------------
    # Camera 1 is the reference frame: extrinsic matrix is [I | 0]
    ref_extrinsic = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0]])
    proj_mat_left = intrinsics_left.dot(ref_extrinsic)   # P1 = K1 * [I | 0]

    # The Essential matrix yields 4 candidate extrinsic matrices for camera 2
    # (two possible rotations × two possible translation directions)
    candidate_extrinsics = helper.camera2(ess_mat)   # Shape: (3, 4, 4)

    # -------------------------------------------------------------------------
    # Step 6: Triangulate each candidate and pick the one with most positive depths
    # -------------------------------------------------------------------------
    # A 3D point is valid only if it is in FRONT of both cameras (positive Z depth).
    # We pick the candidate that maximizes the number of points with positive depth.
    best_extrinsic   = None
    best_proj_right  = None
    best_world_pts   = None
    max_valid_depth  = -1

    for cand_idx in range(4):
        cand_extrinsic  = candidate_extrinsics[:, :, cand_idx]   # One 3x4 [R | t] candidate
        cand_proj_right = intrinsics_right.dot(cand_extrinsic)   # P2 = K2 * [R | t]

        # Triangulate 3D world points from the 2D correspondences
        reconstructed_pts = sub.triangulate(proj_mat_left, query_pts_left,
                                            cand_proj_right, query_pts_right)

        # Count how many reconstructed points have positive Z (in front of camera)
        num_valid = np.sum(reconstructed_pts[:, 2] > 0)
        print(f"Candidate {cand_idx}: {num_valid}/{reconstructed_pts.shape[0]} points with positive depth.")

        # Keep this candidate if it has the most valid (positive-depth) points
        if num_valid > max_valid_depth:
            max_valid_depth  = num_valid
            best_extrinsic   = cand_extrinsic
            best_proj_right  = cand_proj_right
            best_world_pts   = reconstructed_pts

    print("\nSelected Extrinsic Matrix for Camera 2:")
    print(best_extrinsic)

    # Save the extrinsic matrices for potential use in dense reconstruction
    np.savez('../data/extrinsics.npz',
             R1=ref_extrinsic[:, :3],       # Rotation of camera 1 (identity)
             R2=best_extrinsic[:, :3],      # Rotation of camera 2
             t1=ref_extrinsic[:, 3],        # Translation of camera 1 (zero)
             t2=best_extrinsic[:, 3])       # Translation of camera 2

    # -------------------------------------------------------------------------
    # Step 7: Compute reprojection error to evaluate reconstruction quality
    # -------------------------------------------------------------------------
    # Reprojection error = average pixel distance between observed and re-projected points
    _, reprojection_err = compute_reprojection_error(
        proj_mat_left, query_pts_left,
        best_proj_right, query_pts_right,
        best_world_pts
    )
    print(f"\nReprojection Error: {reprojection_err:.4f} pixels")

    # -------------------------------------------------------------------------
    # Step 8: Plot and save the sparse 3D point cloud
    # -------------------------------------------------------------------------
    reconstruction_fig = plt.figure(figsize=(10, 8))
    cloud_axes = reconstruction_fig.add_subplot(111, projection='3d')

    cloud_axes.scatter(
        best_world_pts[:, 0],
        best_world_pts[:, 1],
        best_world_pts[:, 2],
        c='b', marker='.'
    )
    cloud_axes.set_xlabel('X')
    cloud_axes.set_ylabel('Y')
    cloud_axes.set_zlabel('Z')
    cloud_axes.set_title('Sparse 3D Reconstruction')

    # View from above (top-down) to match standard stereo reconstruction convention
    cloud_axes.view_init(elev=-90, azim=-90)

    output_path = '../results_custom_reconstruction.png'
    plt.savefig(output_path)
    print(f"3D reconstruction plot saved to {output_path}")


def compute_reprojection_error(proj_left, observed_left, proj_right, observed_right, world_pts):
    """
    Computes the mean reprojection error across both camera views.

    For each 3D point, it is projected back into image 1 and image 2 using
    the respective projection matrices. The error is the average pixel distance
    between the re-projected and originally observed 2D positions.

        Input:  proj_left      - 3x4 projection matrix for camera 1
                observed_left  - Nx2 observed 2D points in image 1
                proj_right     - 3x4 projection matrix for camera 2
                observed_right - Nx2 observed 2D points in image 2
                world_pts      - Nx3 triangulated 3D world points
        Output: reprojected_left - Nx2 re-projected points in image 1
                mean_pixel_err   - scalar mean reprojection error (in pixels)
    """
    num_pts = world_pts.shape[0]

    # Convert 3D points to homogeneous coordinates [X, Y, Z, 1]
    world_pts_homo = np.hstack((world_pts, np.ones((num_pts, 1))))

    # Project 3D points into each image: P * X_homo → [u*w, v*w, w]
    reprojected_left_homo  = proj_left.dot(world_pts_homo.T).T
    reprojected_right_homo = proj_right.dot(world_pts_homo.T).T

    # Convert from homogeneous to Euclidean by dividing by the third coordinate (w)
    reprojected_left  = reprojected_left_homo[:, :2]  / reprojected_left_homo[:, 2:]
    reprojected_right = reprojected_right_homo[:, :2] / reprojected_right_homo[:, 2:]

    # Compute per-point Euclidean distance between observed and reprojected positions
    error_left  = np.linalg.norm(observed_left  - reprojected_left,  axis=1)
    error_right = np.linalg.norm(observed_right - reprojected_right, axis=1)

    # Mean total error across both views
    mean_pixel_err = np.mean(error_left + error_right)

    return reprojected_left, mean_pixel_err


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Sparse 3D Reconstruction on custom stereo data.')

    parser.add_argument('--im1',        type=str, default='../data/im1.png',
                        help='Path to the left/first image')
    parser.add_argument('--im2',        type=str, default='../data/im2.png',
                        help='Path to the right/second image')
    parser.add_argument('--corresp',    type=str, default='../data/some_corresp.npz',
                        help='Path to known point correspondences .npz file (must contain pts1 and pts2 keys)')
    parser.add_argument('--coords',     type=str, default='../data/temple_coords.npz',
                        help='Path to query coordinates in image 1 .npz file (for triangulation)')
    parser.add_argument('--intrinsics', type=str, default='../data/intrinsics.npz',
                        help='Path to camera intrinsics .npz file (must contain K1 and K2 keys)')

    cli_args = parser.parse_args()

    run_sparse_reconstruction(
        cli_args.im1,
        cli_args.im2,
        cli_args.corresp,
        cli_args.coords,
        cli_args.intrinsics
    )