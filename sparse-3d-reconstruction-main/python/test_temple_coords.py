import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper
import argparse

def test_custom_coords(im1_path, im2_path, corresp_path, coords_path, intrinsics_path):
    # 1. Load the two images and the point correspondences
    im1 = plt.imread(im1_path)
    im2 = plt.imread(im2_path)
    corresp_data = np.load(corresp_path)
    pts1_corresp = corresp_data['pts1']
    pts2_corresp = corresp_data['pts2']
    
    # 2. Run eight_point to compute fundamental matrix F
    M = max(im1.shape[0], im1.shape[1])
    F = sub.eight_point(pts1_corresp, pts2_corresp, M)
    print("Fundamental Matrix F:")
    print(F)
    
    # 3. Load points in image 1 and run epipolar_correspondences
    coords_data = np.load(coords_path)
    # Handle different possible key names for the coordinates
    if 'pts1' in coords_data:
        pts1 = coords_data['pts1']
    elif 'x1' in coords_data and 'y1' in coords_data:
        pts1 = np.hstack((coords_data['x1'], coords_data['y1']))
    else:
        # Fallback: use the first array in the file
        pts1 = coords_data[coords_data.files[0]]
    
    pts1 = pts1.astype(float)
    
    print("\nFinding epipolar correspondences for the target coordinates...")
    pts2 = sub.epipolar_correspondences(im1, im2, F, pts1)
    
    # 4. Load intrinsics and compute essential matrix E
    intrinsics = np.load(intrinsics_path)
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    
    E = sub.essential_matrix(F, K1, K2)
    print("\nEssential Matrix E:")
    print(E)
    
    # 5. Compute first camera projection matrix P1 and use camera2 for candidates
    # P1 is [I | 0] since rotation/translation are zero for camera 1
    M1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    P1 = K1.dot(M1)
    
    M2_candidates = helper.camera2(E)
    
    # 6 & 7. Triangulate candidates, figure out correct P2 based on positive depth
    best_M2 = None
    best_P2 = None
    best_pts3d = None
    max_positive_depth = -1
    
    for i in range(4):
        M2_cand = M2_candidates[:, :, i]
        P2_cand = K2.dot(M2_cand)
        
        pts3d = sub.triangulate(P1, pts1, P2_cand, pts2)
        
        # Check depth (Z coordinate should be positive)
        num_positive = np.sum(pts3d[:, 2] > 0)
        print(f"Candidate {i}: {num_positive}/{pts3d.shape[0]} points with positive depth.")
        
        if num_positive > max_positive_depth:
            max_positive_depth = num_positive
            best_M2 = M2_cand
            best_P2 = P2_cand
            best_pts3d = pts3d
            
    print("\nCorrect Extrinsic Matrix M2:")
    print(best_M2)
    
    # Save the extrinsic matrix for part 2 (dense reconstruction) if needed
    np.savez('../data/extrinsics.npz', R1=M1[:,:3], R2=best_M2[:,:3], t1=M1[:,3], t2=best_M2[:,3])
    
    # Calculate reprojection error
    P_temp, err_opt = get_reprojection_error(P1, pts1, best_P2, pts2, best_pts3d)
    print(f"\nReprojection Error: {err_opt:.4f}")
    
    # 8. Use matplotlib to plot point correspondences in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(best_pts3d[:, 0], best_pts3d[:, 1], best_pts3d[:, 2], c='b', marker='.')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('Sparse 3D Reconstruction')
    
    ax.view_init(elev=-90, azim=-90)
    
    plt.savefig('../results_custom_reconstruction.png')
    print("Plot saved to ../results_custom_reconstruction.png")
    # plt.show()

def get_reprojection_error(P1, pts1, P2, pts2, pts3d):
    pts3d_homo = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    
    pts1_proj = P1.dot(pts3d_homo.T).T
    pts2_proj = P2.dot(pts3d_homo.T).T
    
    pts1_proj = pts1_proj[:, :2] / pts1_proj[:, 2:]
    pts2_proj = pts2_proj[:, :2] / pts2_proj[:, 2:]
    
    err1 = np.linalg.norm(pts1 - pts1_proj, axis=1)
    err2 = np.linalg.norm(pts2 - pts2_proj, axis=1)
    
    mean_error = np.mean(err1 + err2)
    return pts1_proj, mean_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Sparse 3D Reconstruction on custom data.')
    parser.add_argument('--im1', type=str, default='../data/im1.png', help='Path to first image')
    parser.add_argument('--im2', type=str, default='../data/im2.png', help='Path to second image')
    parser.add_argument('--corresp', type=str, default='../data/some_corresp.npz', help='Path to known point correspondences npz file (must contain pts1 and pts2 keys)')
    parser.add_argument('--coords', type=str, default='../data/temple_coords.npz', help='Path to target coordinates to triangulate in image 1 npz file')
    parser.add_argument('--intrinsics', type=str, default='../data/intrinsics.npz', help='Path to camera intrinsics npz file (must contain K1 and K2 keys)')
    
    args = parser.parse_args()
    
    test_custom_coords(args.im1, args.im2, args.corresp, args.coords, args.intrinsics)
