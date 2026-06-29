import numpy as np
from helper import refineF
from scipy.ndimage import gaussian_filter

def eight_point(pts1, pts2, M):
    """
    Q2.1: Eight Point Algorithm
        Input:  pts1, Nx2 Matrix
                pts2, Nx2 Matrix
                M, a scalar parameter computed as max (imwidth, imheight)
        Output: F, the fundamental matrix
    """
    pts1_scaled = pts1 / M
    pts2_scaled = pts2 / M

    N = pts1_scaled.shape[0]
    A = np.zeros((N, 9))

    for i in range(N):
        x1, y1 = pts1_scaled[i]
        x2, y2 = pts2_scaled[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    u, s, vh = np.linalg.svd(A)
    F = vh[-1].reshape(3, 3)

    # refineF enforces rank 2 constraint internally and optimizes
    F = refineF(F, pts1_scaled, pts2_scaled)

    T = np.diag([1/M, 1/M, 1])
    unscaled_F = T.T.dot(F).dot(T)

    return unscaled_F

def epipolar_correspondences(im1, im2, F, pts1):
    """
    Q2.2: Find epipolar correspondences
        Input:  im1, the first image
                im2, the second image
                F, the fundamental matrix
                pts1, Nx2 Matrix of points in image 1
        Output: pts2, Nx2 Matrix of corresponding points in image 2
    """
    kernel_size = 21 # Window size
    sigma = 5
    
    # Gaussian window
    window = np.zeros((kernel_size, kernel_size))
    window[kernel_size//2, kernel_size//2] = 1
    kernel = gaussian_filter(window, sigma)
    kernel /= np.sum(kernel)
    kernel = np.dstack((kernel, kernel, kernel))
    
    sy, sx, _ = im2.shape
    k_half = kernel_size // 2
    
    pts2 = np.zeros_like(pts1)
    
    for idx in range(pts1.shape[0]):
        x1, y1 = pts1[idx]
        xc = int(np.round(x1))
        yc = int(np.round(y1))
        
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2 + l[1]**2)
        if s == 0:
            continue
            
        l = l / s
        
        if l[0] != 0:
            ye = sy - 1
            ys = 0
            xe = -(l[1] * ye + l[2]) / l[0]
            xs = -(l[1] * ys + l[2]) / l[0]
        else:
            xe = sx - 1
            xs = 0
            ye = -(l[0] * xe + l[2]) / l[1]
            ys = -(l[0] * xs + l[2]) / l[1]
            
        N_points = int(max(abs(ye-ys), abs(xe-xs)))
        if N_points == 0:
            N_points = 1
            
        x2_list = np.linspace(xs, xe, N_points)
        y2_list = np.linspace(ys, ye, N_points)
        
        x2_list = np.rint(x2_list).astype(int)
        y2_list = np.rint(y2_list).astype(int)
        
        min_error = np.inf
        best_x2, best_y2 = xc, yc
        
        if yc >= k_half and xc >= k_half and yc + k_half + 1 <= sy and xc + k_half + 1 <= sx:
            patch_1 = im1[yc - k_half: yc + k_half + 1, xc - k_half: xc + k_half + 1, :].astype(float)
            
            for i in range(len(x2_list)):
                x2 = x2_list[i]
                y2 = y2_list[i]
                
                # Heuristic: stereo images are usually somewhat aligned horizontally
                if abs(y2 - yc) > 40:
                    continue
                    
                if y2 >= k_half and x2 >= k_half and y2 + k_half + 1 <= sy and x2 + k_half + 1 <= sx:
                    patch_2 = im2[y2 - k_half: y2 + k_half + 1, x2 - k_half: x2 + k_half + 1, :].astype(float)
                    
                    diff = patch_1 - patch_2
                    diff_gaussian = np.multiply(kernel, diff)
                    err = np.linalg.norm(diff_gaussian)
                    
                    if err < min_error:
                        min_error = err
                        best_x2 = x2
                        best_y2 = y2
                        
        pts2[idx] = [best_x2, best_y2]
        
    return pts2

def essential_matrix(F, K1, K2):
    """
    Q2.3: Compute the essential matrix E.
        Input:  F, fundamental matrix
                K1, internal camera calibration matrix of camera 1
                K2, internal camera calibration matrix of camera 2
        Output: E, the essential matrix
    """
    E = K2.T.dot(F).dot(K1)
    return E

def triangulate(P1, pts1, P2, pts2):
    """
    Q2.4: Triangulate a set of 2D coordinates in the image to a set of 3D points.
        Input:  P1, the 3x4 camera matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                P2, the 3x4 camera matrix
                pts2, the Nx2 matrix with the 2D image coordinates per row
        Output: pts3d, the Nx3 matrix with the corresponding 3D points per row
    """
    pts3d = []
    
    for i in range(pts1.shape[0]):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        
        A = np.array([
            y1 * P1[2, :] - P1[1, :],
            P1[0, :] - x1 * P1[2, :],
            y2 * P2[2, :] - P2[1, :],
            P2[0, :] - x2 * P2[2, :]
        ])
        
        u, s, vh = np.linalg.svd(A)
        X = vh[-1]
        X = X / X[-1]
        pts3d.append(X[:3])
        
    return np.array(pts3d)
