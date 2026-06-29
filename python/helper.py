import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

def _enforce_rank2(matrix):
    """
    Forces a 3x3 matrix to be rank-2 by zeroing out the smallest singular value.
    This is required for a valid Fundamental matrix (which must have rank 2).
    """
    # Decompose the matrix into U, singular values, and V using SVD
    left_vecs, singular_vals, right_vecs = np.linalg.svd(matrix)

    # Zero out the last (smallest) singular value to enforce rank-2 constraint
    singular_vals[-1] = 0

    # Reconstruct the matrix from modified singular values
    rank2_matrix = left_vecs.dot(np.diag(singular_vals).dot(right_vecs))
    return rank2_matrix


def _compute_sampson_error(flat_matrix, source_pts, target_pts):
    """
    Computes the Sampson distance (first-order geometric error) between
    point correspondences given a candidate Fundamental matrix.

    This is used as the optimization objective when refining F.
    The Sampson error is a good approximation of reprojection error
    and is efficient to compute.
    """
    # Reshape flat vector back into 3x3 matrix and enforce rank-2
    fund_matrix = _enforce_rank2(flat_matrix.reshape([3, 3]))

    total_correspondences = source_pts.shape[0]

    # Convert 2D points to homogeneous coordinates by appending a column of 1s
    homo_source = np.concatenate([source_pts, np.ones([total_correspondences, 1])], axis=1)
    homo_target = np.concatenate([target_pts, np.ones([total_correspondences, 1])], axis=1)

    # Compute epilines for source points: F * p1
    epilines_from_source = fund_matrix.dot(homo_source.T)

    # Compute epilines for target points using transpose: F^T * p2
    epilines_from_target = fund_matrix.T.dot(homo_target.T)

    total_error = 0
    for line_src, line_tgt, h_tgt in zip(epilines_from_source.T, epilines_from_target.T, homo_target):
        # Denominators: sum of squares of the first two coordinates of each epiline
        # These represent the squared norms of the line direction vectors
        denom_src = line_src[0]**2 + line_src[1]**2
        denom_tgt = line_tgt[0]**2 + line_tgt[1]**2

        # Skip degenerate cases where the line direction is zero (avoid division by zero)
        if denom_src == 0 or denom_tgt == 0:
            continue

        # Sampson error formula: (p2^T * F * p1)^2 * (1/||Fp1||^2 + 1/||F^Tp2||^2)
        # h_tgt.dot(line_src) gives the algebraic distance of p2 from the epiline
        total_error += (h_tgt.dot(line_src))**2 * (1 / denom_src + 1 / denom_tgt)

    return total_error


def refineF(initial_F, source_pts, target_pts):
    """
    Refines a Fundamental matrix using Powell's numerical optimization method.
    Minimizes the Sampson error over all point correspondences.

    Args:
        initial_F   : Initial 3x3 Fundamental matrix estimate
        source_pts  : Nx2 array of points from image 1
        target_pts  : Nx2 array of corresponding points from image 2

    Returns:
        Refined 3x3 Fundamental matrix (rank-2 enforced)
    """
    # Flatten F to a 1D vector for the optimizer, then optimize using Powell's method
    optimized_flat = scipy.optimize.fmin_powell(
        lambda vec: _compute_sampson_error(vec, source_pts, target_pts),
        initial_F.reshape([-1]),   # Initial guess as flat vector
        maxiter=10000,             # Max number of iterations allowed
        maxfun=10000,              # Max number of function evaluations
        disp=False                 # Suppress console output
    )

    # Reshape back to 3x3 and enforce the rank-2 constraint before returning
    return _enforce_rank2(optimized_flat.reshape([3, 3]))


def camera2(essential_mat):
    """
    Recovers the four possible camera projection matrices (M2) from
    an Essential matrix using SVD decomposition.

    The Essential matrix encodes rotation and translation between two cameras.
    There are 4 possible solutions; the correct one is chosen later via
    triangulation (cheirality check).

    Args:
        essential_mat : 3x3 Essential matrix

    Returns:
        A 3x4x4 array containing 4 candidate projection matrices
    """
    # SVD decomposition of the Essential matrix
    left_vecs, sing_vals, right_vecs = np.linalg.svd(essential_mat)

    # Enforce the ideal singular value structure [s, s, 0] for an Essential matrix
    avg_singular = sing_vals[:2].mean()
    essential_mat = left_vecs.dot(
        np.array([[avg_singular, 0, 0],
                  [0, avg_singular, 0],
                  [0, 0,           0]])
    ).dot(right_vecs)

    # Recompute SVD after correction
    left_vecs, sing_vals, right_vecs = np.linalg.svd(essential_mat)

    # W matrix used to extract rotation from the SVD of E
    # It encodes a 90-degree rotation in the plane
    rot_helper = np.array([[0, -1, 0],
                           [1,  0, 0],
                           [0,  0, 1]])

    # Ensure the recovered rotation has a positive determinant (i.e., is a valid rotation matrix)
    if np.linalg.det(left_vecs.dot(rot_helper).dot(right_vecs)) < 0:
        rot_helper = -rot_helper

    # Normalize the translation vector using the max absolute value of its components
    translation_col = left_vecs[:, 2].reshape([-1, 1])
    norm_translation = translation_col / abs(translation_col).max()

    # Build 4 candidate [R | t] projection matrices (3x4), stored along axis=2
    # Combinations: (W or W^T) x (+t or -t)
    candidate_projections = np.zeros([3, 4, 4])

    rotation_W  = left_vecs.dot(rot_helper).dot(right_vecs)
    rotation_WT = left_vecs.dot(rot_helper.T).dot(right_vecs)

    candidate_projections[:, :, 0] = np.concatenate([rotation_W,   norm_translation],  axis=1)
    candidate_projections[:, :, 1] = np.concatenate([rotation_W,  -norm_translation],  axis=1)
    candidate_projections[:, :, 2] = np.concatenate([rotation_WT,  norm_translation],  axis=1)
    candidate_projections[:, :, 3] = np.concatenate([rotation_WT, -norm_translation],  axis=1)

    return candidate_projections


def _compute_epipoles(fund_matrix):
    """
    Computes the two epipoles from a Fundamental matrix.

    The epipole e1 is the null space of F   (right null vector → F * e1 = 0)
    The epipole e2 is the null space of F^T (left  null vector → F^T * e2 = 0)

    Both are found as the last row of V in the SVD, which corresponds
    to the smallest singular value (effectively zero for a rank-2 matrix).
    """
    # Epipole in image 1: right null space of F
    _, _, right_vecs_F = np.linalg.svd(fund_matrix)
    epipole_img1 = right_vecs_F[-1, :]

    # Epipole in image 2: right null space of F^T
    _, _, right_vecs_FT = np.linalg.svd(fund_matrix.T)
    epipole_img2 = right_vecs_FT[-1, :]

    return epipole_img1, epipole_img2


def displayEpipolarF(img_left, img_right, fund_matrix):
    """
    Interactive visualization of epipolar geometry.

    Click a point in the left image → draws the corresponding epipolar line
    in the right image. Any matching point in image 2 must lie on that line
    (this is the epipolar constraint).

    Args:
        img_left     : Left image (H x W x 3 numpy array)
        img_right    : Right image (H x W x 3 numpy array)
        fund_matrix  : 3x3 Fundamental matrix relating the two images
    """
    # Compute epipoles (not drawn here but available for debugging)
    ep_left, ep_right = _compute_epipoles(fund_matrix)

    # Get right image dimensions for clipping epipolar line endpoints
    img_height, img_width, _ = img_right.shape

    # Set up side-by-side figure with left and right image axes
    fig, [panel_left, panel_right] = plt.subplots(1, 2, figsize=(12, 9))

    panel_left.imshow(img_left)
    panel_left.set_title('Select a point in this image')
    panel_left.set_axis_off()

    panel_right.imshow(img_right)
    panel_right.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    panel_right.set_axis_off()

    print("Click on the left image to see the epipolar line on the right image. Close window when done.")

    try:
        while True:
            # Wait for a single click on the left image; stop on right-click (mouse_stop=2)
            plt.sca(panel_left)
            clicked_pts = plt.ginput(1, mouse_stop=2)

            # Exit loop if no point was selected (user closed or right-clicked)
            if not clicked_pts:
                break

            click_x, click_y = clicked_pts[0]

            # Convert clicked point to homogeneous coordinates
            homo_click = np.array([click_x, click_y, 1])

            # Compute the epipolar line in the right image: l = F * p
            epipolar_line = fund_matrix.dot(homo_click)

            # Normalize the line by its direction norm so distances are meaningful
            line_norm = np.sqrt(epipolar_line[0]**2 + epipolar_line[1]**2)
            if line_norm == 0:
                print('Zero line vector in displayEpipolar')
                continue
            epipolar_line = epipolar_line / line_norm

            # Find two endpoints of the epipolar line clipped to the image boundaries
            # Line equation: l[0]*x + l[1]*y + l[2] = 0
            if epipolar_line[0] != 0:
                # Solve for x at y=0 (top) and y=height-1 (bottom)
                y_top    = 0
                y_bottom = img_height - 1
                x_bottom = -(epipolar_line[1] * y_bottom + epipolar_line[2]) / epipolar_line[0]
                x_top    = -(epipolar_line[1] * y_top    + epipolar_line[2]) / epipolar_line[0]
            else:
                # If l[0]==0, line is horizontal → solve for y at x=0 and x=width-1
                x_left  = 0
                x_right = img_width - 1
                y_right = -(epipolar_line[0] * x_right + epipolar_line[2]) / epipolar_line[1]
                y_left  = -(epipolar_line[0] * x_left  + epipolar_line[2]) / epipolar_line[1]
                x_bottom, x_top = x_right, x_left
                y_bottom, y_top = y_right, y_left

            # Mark the clicked point on the left image with a red star
            panel_left.plot(click_x, click_y, '*', markersize=6, linewidth=2, color='r')

            # Draw the epipolar line on the right image
            panel_right.plot([x_top, x_bottom], [y_top, y_bottom], linewidth=2)

            # Refresh the figure to show updates
            plt.draw()

    except Exception:
        # Silently catch any matplotlib/interaction errors (e.g., window closed mid-click)
        pass