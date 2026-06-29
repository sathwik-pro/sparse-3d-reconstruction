import numpy as np
from helper import refineF
from scipy.ndimage import gaussian_filter


def eight_point(source_pts, target_pts, scale_factor):
    """
    Eight Point Algorithm for computing the Fundamental Matrix.
        Input:  source_pts   - Nx2 matrix of points from image 1
                target_pts   - Nx2 matrix of points from image 2
                scale_factor - normalization scalar = max(image_width, image_height)
        Output: fund_matrix  - 3x3 Fundamental matrix
    """
    # --- Step 1: Normalize points by scale factor ---
    # Dividing by M brings coordinates into [0, 1] range,
    # which improves numerical stability during SVD
    norm_src = source_pts / scale_factor
    norm_tgt = target_pts / scale_factor

    num_pts = norm_src.shape[0]

    # --- Step 2: Build the constraint matrix A ---
    # Each point correspondence (x1,y1) <-> (x2,y2) contributes one row:
    # [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    # The fundamental matrix F satisfies: p2^T * F * p1 = 0
    constraint_mat = np.zeros((num_pts, 9))

    for row_idx in range(num_pts):
        src_x, src_y = norm_src[row_idx]
        tgt_x, tgt_y = norm_tgt[row_idx]
        constraint_mat[row_idx] = [
            tgt_x * src_x,   # x2 * x1
            tgt_x * src_y,   # x2 * y1
            tgt_x,           # x2
            tgt_y * src_x,   # y2 * x1
            tgt_y * src_y,   # y2 * y1
            tgt_y,           # y2
            src_x,           # x1
            src_y,           # y1
            1                # constant term
        ]

    # --- Step 3: Solve using SVD ---
    # The solution for F is the last row of Vh (smallest singular value),
    # which minimizes ||A * f||^2 subject to ||f|| = 1
    left_vecs, sing_vals, right_vecs_T = np.linalg.svd(constraint_mat)

    # Reshape the last row of Vh into a 3x3 matrix — this is our initial F estimate
    initial_fund_mat = right_vecs_T[-1].reshape(3, 3)

    # --- Step 4: Refine F ---
    # refineF enforces the rank-2 constraint (via SVD zeroing) and minimizes
    # the Sampson error using Powell's optimization method
    refined_fund_mat = refineF(initial_fund_mat, norm_src, norm_tgt)

    # --- Step 5: Denormalize F ---
    # Since we normalized points with T = diag(1/M, 1/M, 1),
    # the true F in original coordinates is: F_real = T^T * F_norm * T
    denorm_transform = np.diag([1 / scale_factor, 1 / scale_factor, 1])
    final_fund_mat = denorm_transform.T.dot(refined_fund_mat).dot(denorm_transform)

    return final_fund_mat


def epipolar_correspondences(img_src, img_tgt, fund_matrix, src_pts):
    """
    Find corresponding points in image 2 for given points in image 1,
    by searching along epipolar lines using patch-based matching.

        Input:  img_src     - First image (H x W x 3)
                img_tgt     - Second image (H x W x 3)
                fund_matrix - 3x3 Fundamental matrix
                src_pts     - Nx2 matrix of points in image 1
        Output: matched_pts - Nx2 matrix of corresponding points in image 2
    """
    # --- Gaussian patch kernel setup ---
    # We use a weighted patch window so that pixels near the center
    # of the patch contribute more to the similarity score than edge pixels
    patch_size  = 21          # Total window width/height in pixels
    blur_sigma  = 5           # Standard deviation for Gaussian blur
    half_patch  = patch_size // 2

    # Create a Gaussian kernel: place a single 1 at center and blur it
    raw_window  = np.zeros((patch_size, patch_size))
    raw_window[half_patch, half_patch] = 1
    gauss_kernel = gaussian_filter(raw_window, blur_sigma)
    gauss_kernel /= np.sum(gauss_kernel)   # Normalize so weights sum to 1

    # Stack kernel across 3 color channels for element-wise multiplication
    gauss_kernel_3ch = np.dstack((gauss_kernel, gauss_kernel, gauss_kernel))

    tgt_height, tgt_width, _ = img_tgt.shape

    # Output array to store the best matching point in image 2 for each src point
    matched_pts = np.zeros_like(src_pts)

    # --- Search for each source point's correspondence along its epipolar line ---
    for pt_idx in range(src_pts.shape[0]):
        px, py = src_pts[pt_idx]

        # Round to nearest integer pixel coordinates
        px_int = int(np.round(px))
        py_int = int(np.round(py))

        # --- Compute the epipolar line in image 2 ---
        # For a point p1 in image 1, the epipolar line in image 2 is: l = F * p1
        homo_pt = np.array([px_int, py_int, 1])
        epi_line = fund_matrix.dot(homo_pt)

        # Normalize the line vector by its direction magnitude (first two components)
        # so that distances to the line are in pixel units
        line_magnitude = np.sqrt(epi_line[0]**2 + epi_line[1]**2)
        if line_magnitude == 0:
            continue   # Degenerate case: skip this point
        epi_line = epi_line / line_magnitude

        # --- Find the two endpoints of the epipolar line clipped to image 2 bounds ---
        # Line equation: l[0]*x + l[1]*y + l[2] = 0
        if epi_line[0] != 0:
            # Line has a non-zero x-component → solve for x at y=0 and y=height-1
            y_top    = 0
            y_bottom = tgt_height - 1
            x_bottom = -(epi_line[1] * y_bottom + epi_line[2]) / epi_line[0]
            x_top    = -(epi_line[1] * y_top    + epi_line[2]) / epi_line[0]
        else:
            # Horizontal line → solve for y at x=0 and x=width-1
            x_top    = 0
            x_bottom = tgt_width - 1
            y_bottom = -(epi_line[0] * x_bottom + epi_line[2]) / epi_line[1]
            y_top    = -(epi_line[0] * x_top    + epi_line[2]) / epi_line[1]

        # --- Sample candidate pixels along the epipolar line ---
        # Number of samples = length of the line in pixels (max of dx or dy)
        num_candidates = int(max(abs(y_bottom - y_top), abs(x_bottom - x_top)))
        if num_candidates == 0:
            num_candidates = 1

        candidate_xs = np.rint(np.linspace(x_top, x_bottom, num_candidates)).astype(int)
        candidate_ys = np.rint(np.linspace(y_top, y_bottom, num_candidates)).astype(int)

        # --- Patch matching: find the candidate point with the lowest error ---
        min_patch_error = np.inf
        best_match_x    = px_int
        best_match_y    = py_int

        # Extract the reference patch from image 1 (centered at clicked point)
        # Only proceed if the patch fits within image boundaries
        src_in_bounds = (
            py_int >= half_patch and px_int >= half_patch and
            py_int + half_patch + 1 <= tgt_height and
            px_int + half_patch + 1 <= tgt_width
        )

        if src_in_bounds:
            ref_patch = img_src[
                py_int - half_patch : py_int + half_patch + 1,
                px_int - half_patch : px_int + half_patch + 1,
                :
            ].astype(float)

            # Evaluate each candidate point along the epipolar line
            for cand_idx in range(len(candidate_xs)):
                cand_x = candidate_xs[cand_idx]
                cand_y = candidate_ys[cand_idx]

                # Heuristic: stereo images are roughly horizontally aligned,
                # so skip candidates that are too far vertically from the source point
                if abs(cand_y - py_int) > 40:
                    continue

                # Check that the candidate patch fits within image 2 boundaries
                tgt_in_bounds = (
                    cand_y >= half_patch and cand_x >= half_patch and
                    cand_y + half_patch + 1 <= tgt_height and
                    cand_x + half_patch + 1 <= tgt_width
                )

                if tgt_in_bounds:
                    # Extract the candidate patch from image 2
                    cand_patch = img_tgt[
                        cand_y - half_patch : cand_y + half_patch + 1,
                        cand_x - half_patch : cand_x + half_patch + 1,
                        :
                    ].astype(float)

                    # Compute Gaussian-weighted pixel difference between patches
                    pixel_diff          = ref_patch - cand_patch
                    weighted_diff       = np.multiply(gauss_kernel_3ch, pixel_diff)
                    patch_error         = np.linalg.norm(weighted_diff)

                    # Keep track of the best (lowest error) match
                    if patch_error < min_patch_error:
                        min_patch_error = patch_error
                        best_match_x    = cand_x
                        best_match_y    = cand_y

        matched_pts[pt_idx] = [best_match_x, best_match_y]

    return matched_pts


def essential_matrix(fund_matrix, intrinsics_cam1, intrinsics_cam2):
    """
    Compute the Essential Matrix from the Fundamental Matrix and camera intrinsics.
        Input:  fund_matrix     - 3x3 Fundamental matrix F
                intrinsics_cam1 - 3x3 calibration matrix K1 of camera 1
                intrinsics_cam2 - 3x3 calibration matrix K2 of camera 2
        Output: ess_matrix      - 3x3 Essential matrix E

    Relationship: E = K2^T * F * K1
    The Essential matrix encodes rotation and translation between cameras,
    but in normalized (calibrated) image coordinates rather than pixel coordinates.
    """
    ess_matrix = intrinsics_cam2.T.dot(fund_matrix).dot(intrinsics_cam1)
    return ess_matrix


def triangulate(proj_mat1, src_pts, proj_mat2, tgt_pts):
    """
    Triangulate 3D world points from 2D correspondences across two views.
        Input:  proj_mat1 - 3x4 projection matrix of camera 1
                src_pts   - Nx2 matrix of 2D points in image 1
                proj_mat2 - 3x4 projection matrix of camera 2
                tgt_pts   - Nx2 matrix of 2D points in image 2
        Output: world_pts - Nx3 matrix of reconstructed 3D points

    Uses the Direct Linear Transform (DLT): for each correspondence,
    builds a 4x4 system A*X=0 and solves via SVD.
    """
    world_pts = []

    for pt_idx in range(src_pts.shape[0]):
        src_x, src_y = src_pts[pt_idx]
        tgt_x, tgt_y = tgt_pts[pt_idx]

        # --- Build the DLT constraint matrix A ---
        # Each 2D point gives 2 linear constraints on the 3D point X:
        #   (y * P[2,:] - P[1,:]) * X = 0   ← from y-coordinate
        #   (P[0,:] - x * P[2,:]) * X = 0   ← from x-coordinate
        # Stacking both images gives a 4x4 system: A * X = 0
        dlt_mat = np.array([
            src_y * proj_mat1[2, :] - proj_mat1[1, :],   # y1 constraint from cam1
            proj_mat1[0, :] - src_x * proj_mat1[2, :],   # x1 constraint from cam1
            tgt_y * proj_mat2[2, :] - proj_mat2[1, :],   # y2 constraint from cam2
            proj_mat2[0, :] - tgt_x * proj_mat2[2, :]    # x2 constraint from cam2
        ])

        # --- Solve A * X = 0 using SVD ---
        # The solution X is the last row of Vh (corresponds to the smallest singular value)
        _, _, right_vecs_T = np.linalg.svd(dlt_mat)
        homo_3d_pt = right_vecs_T[-1]

        # Convert from homogeneous [X, Y, Z, W] to Euclidean [X/W, Y/W, Z/W]
        homo_3d_pt = homo_3d_pt / homo_3d_pt[-1]
        world_pts.append(homo_3d_pt[:3])

    return np.array(world_pts)