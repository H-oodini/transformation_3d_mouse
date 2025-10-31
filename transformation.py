"""
Filename: transformation.py
Author: Angelina Besser

Description:
This file is a fork of Jazemins Project 3d_transformation. 
It uses the same folder structure for all input and output paths.

Note:
This is legacy code. Stereo calibration via chessboard has not been 
implemented yet, so the current intrinsics and rectification setup are
still based on assumptions rather than a proper calibration.

The script performs 3D reconstruction using two projection matrices because
the left and right camera images have different resolutions.

"""

import numpy as np
import cv2
import open3d as o3d
from typing import List, Dict, Tuple
from ultralytics import YOLO
from pathlib import Path


# ---------------- YOLO detection ----------------
def detect_features(model:YOLO, image_path:str):
    """
    Detect objects in an image using a YOLO model.
    """
    # Run inference on the given image path; returns a list of results objects
    results = model(image_path)
    detections = []
    # Iterate over detected bounding boxes in the first result (single image case)
    for box in results[0].boxes:
        # Class index and name for this detection
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        # xyxy coordinates (top-left and bottom-right corners)
        x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
        # Compute bbox center in pixel coordinates
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        # Collect a compact dict describing this detection
        detections.append({
            'class': cls_name,
            'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
            'cx': float(cx),
            'cy': float(cy)
        })
    # Returns a list of dicts: class name, bbox tuple, and center (cx, cy)
    return detections

# ---------------- SIFT detection ----------------
def detect_sift_keypoints(image_path:str, mask:np.ndarray=None):
    """
    Find SIFT keypoints and descriptors in an image (optionally masked).
    """
    # Load image in grayscale for SIFT
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Create SIFT detector-extractor
    sift = cv2.SIFT_create()
    # Detect keypoints and compute descriptors; mask can restrict detection region
    keypoints, descriptors = sift.detectAndCompute(img, mask)
    # Returns: list of cv2.KeyPoint and a descriptor array (N x 128) or None
    return keypoints, descriptors

def match_sift_features(desc1, desc2, ratio_thresh:float=0.75):
    """
    Match SIFT descriptors between two images using Lowe's ratio test.
    """
    # Brute-force matcher with default L2 for SIFT
    bf = cv2.BFMatcher()
    # For each descriptor in desc1, find the two nearest matches in desc2
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for pair in matches:
        # Some pairs can be incomplete; skip those
        if len(pair) != 2:
            continue
        m, n = pair
        # Lowe's ratio test: keep only sufficiently distinctive matches
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    # Returns a filtered list of cv2.DMatch objects
    return good_matches

def sift_matches_full_image(left_img: str, right_img: str, ratio_thresh: float = 0.75):
    # Load both images as grayscale; full-image matching
    imgL = cv2.imread(left_img, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE)
    # Build SIFT extractor
    sift = cv2.SIFT_create()
    # Detect and compute SIFT features for both images
    kp1, desc1 = sift.detectAndCompute(imgL, None)
    kp2, desc2 = sift.detectAndCompute(imgR, None)
    # If no descriptors found in either image, nothing to match
    if desc1 is None or desc2 is None:
        return []

    # KNN matching between the two descriptor sets
    bf = cv2.BFMatcher()
    knn = bf.knnMatch(desc1, desc2, k=2)
    filtered_matches = []
    for pair in knn:
        # Skip incomplete pairs to avoid index errors
        if len(pair) != 2:
            continue
        m, n = pair
        # Standard Lowe ratio for robustness
        if m.distance < ratio_thresh * n.distance:
            filtered_matches.append(m)

    # Extract matched coordinate pairs for robust geometry filtering
    ptsL = np.float32([kp1[m.queryIdx].pt for m in filtered_matches])
    ptsR = np.float32([kp2[m.trainIdx].pt for m in filtered_matches])
    if len(ptsL) >= 8:
        # Estimate fundamental matrix with RANSAC to reject outliers
        F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_RANSAC, 1.0, 0.99)
        if mask is not None:
            mask = mask.ravel().astype(bool)
            ptsL, ptsR = ptsL[mask], ptsR[mask]
        # in dein altes Dict-Format bringen
    # Convert surviving pairs into the script's expected dict-pair format
    sift_matches = [({'class': 'Scene', 'cx': float(xL), 'cy': float(yL)},
                      {'class': 'Scene', 'cx': float(xR), 'cy': float(yR)})
                     for (xL, yL), (xR, yR) in zip(ptsL, ptsR)]
    # Returns a list of tuple(dict, dict) for left/right points
    return sift_matches

def triangulate_with_proj(matches: List[Tuple[Dict,Dict]], P1: np.ndarray, P2: np.ndarray):
    """
    Triangulate 3D points using projection matrices.
    """
    # Collect left/right 2D points and class labels
    pts_left = []
    pts_right = []
    classes = []
    for L, R in matches:
        pts_left.append([L['cx'], L['cy']])
        pts_right.append([R['cx'], R['cy']])
        classes.append(L['class'])
    # If no correspondences, nothing to triangulate
    if not pts_left:
        return []
    # Convert to the shape expected by cv2.triangulatePoints: 2xN arrays
    pts_left = np.array(pts_left).T.astype(np.float64)   # 2xN
    pts_right = np.array(pts_right).T.astype(np.float64) # 2xN
    # Homogeneous linear triangulation using the two projection matrices
    hom_pts = cv2.triangulatePoints(P1, P2, pts_left, pts_right)
    # Dehomogenize: divide by w component to get 3D coordinates
    hom_pts /= hom_pts[3:4, :]
    pts3d = hom_pts[:3, :].T
    # Return list of tuples (x, y, z, class)
    return [(float(x), float(y), float(z), cls) for (x, y, z), cls in zip(pts3d, classes)]

def save_mouse_mesh(pts3d:list, filename:str):
    """
    Save 3D points as a colored mesh of a mouse using Poisson reconstruction.
    """
    # Simple color map per class; default gray if class key not found
    cmap = {'body': [200, 100, 100], 'ear': [200, 200, 50], 'tail': [100, 200, 200]}
    points, colors = [], []
    # Build arrays of 3D points and per-point RGB colors in [0,1]
    for x, y, z, cls in pts3d:
        points.append([x, y, z])
        c = cmap.get(cls.lower(), cmap.get(cls, [180, 180, 180]))  # be tolerant about case
        colors.append([c[0]/255.0, c[1]/255.0, c[2]/255.0])

    # If no points were provided, nothing to save
    if not points:
        print('No points to save')
        return

    # Create an Open3D point cloud and estimate normals for surface reconstruction
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    # Estimate normals for Poisson reconstruction; parameters control neighborhood size
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    # Poisson surface reconstruction; depth controls octree resolution
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    # Crop mesh to the axis-aligned bounding box of the original points to remove shells
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
    # Write mesh to disk (format inferred from filename extension)
    o3d.io.write_triangle_mesh(filename, mesh)
    print(f"Wrote mesh to {filename}")

# utilities to build intrinsics for each image size from native/sensor intrinsics
def make_K_from_native(W0, H0, fx0, fy0, cx0, cy0, W, H, crop_x=0.0, crop_y=0.0):
    """
    Scale native intrinsics (defined at sensor/native size W0xH0) to a target image size WxH.
    Optionally shift principal point if a crop (in pixels at the target size) is applied.
    """
    # Compute scaling from native sensor resolution to the target image resolution
    sx, sy = W / W0, H / H0
    # Scale focal lengths accordingly
    fx = fx0 * sx
    fy = fy0 * sy
    # Shift principal point for scaling and any pixel-space crop
    cx = cx0 * sx - crop_x
    cy = cy0 * sy - crop_y
    # Return a standard 3x3 camera intrinsic matrix
    return np.array([[fx, 0,  cx],
                     [0,  fy, cy],
                     [0,  0,  1]], dtype=np.float64)
    
def compute_depth_png(
    imgL, imgR,
    K_left, K_right,
    R, T,
    baseline_m,
    out_path="depth_map.png",
    num_disp=16*16,   # multiple of 16; raise if scene is very near
    block=5,          # 3/5/7; 5 is a decent default
    use_clahe=True,
):
    """
    Minimal dense depth from a stereo pair.
    Assumes imgL/imgR are already grayscale uint8 (loaded with IMREAD_GRAYSCALE)
    and K_left/K_right reflect the final cropped image sizes.
    Writes a normalized depth PNG to `out_path`. Returns nothing.
    """

    # 0) Sanity: force uint8 grayscale (no color conversions here)
    if imgL is None or imgR is None:
        raise ValueError("Left or right image is None.")
    print(imgL.shape)
    print(imgR.shape)
    # Ensure arrays are 2D uint8 for SGBM
    L = np.squeeze(imgL).astype(np.uint8)
    Rg = np.squeeze(imgR).astype(np.uint8)
    h, w = L.shape

    # 1) Rectify to left size; assume zero distortion
    # Distortion vectors set to zero; rectification based on provided K, R, T
    D1 = np.zeros(5, dtype=np.float64)
    D2 = np.zeros(5, dtype=np.float64)
    # Compute rectification transforms and new projection matrices
    R1, R2, P1r, P2r, Q, roi1, roi2 = cv2.stereoRectify(
        K_left, D1, K_right, D2, (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    # Precompute rectification maps
    m1x, m1y = cv2.initUndistortRectifyMap(K_left,  D1, R1, P1r, (w, h), cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(K_right, D2, R2, P2r, (w, h), cv2.CV_32FC1)
    # Apply rectification
    Lr = cv2.remap(L,  m1x, m1y, cv2.INTER_LINEAR)
    Rr = cv2.remap(Rg, m2x, m2y, cv2.INTER_LINEAR)

    # 2) Crop to common valid ROI (avoids warped borders killing matches)
    # Compute intersection of valid ROIs to maintain correspondence after warping
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    if xb > xa and yb > ya:
        Lr = Lr[ya:yb, xa:xb]
        Rr = Rr[ya:yb, xa:xb]

    # 3) Optional contrast boost for low-texture scenes
    if use_clahe:
        # Apply CLAHE to both images to improve local contrast for matching
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        Lr = clahe.apply(Lr)
        Rr = clahe.apply(Rr)

    # 4) Disparity (simple, but not tiny)
    # Enforce num_disp multiple of 16 for SGBM and keep it non-trivial
    num_disp = int(max(16*6, (num_disp // 16) * 16))  # enforce multiple of 16
    min_disp = 0
    # Configure SGBM; cost parameters P1/P2 follow common heuristics based on block size
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=-1,          # relax LR check to keep more pixels
        uniquenessRatio=5,         # less strict, fewer false rejections
        speckleWindowSize=0, speckleRange=0,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    # Compute disparity in pixels; divide by 16 to undo fixed-point scaling
    disp = sgbm.compute(Lr, Rr).astype(np.float32) / 16.0
    # Mask out invalid disparities
    disp[disp <= (min_disp - 0.5)] = np.nan

    # 5) Depth: Z = f * B / d (use focal from rectified left projection)
    f_px = float(P1r[0, 0])
    depth = np.full_like(disp, np.nan, dtype=np.float32)
    valid = np.isfinite(disp)
    if np.any(valid):
        # Convert disparity to depth in meters using known baseline
        depth[valid] = f_px * float(baseline_m) / disp[valid]

    # 6) Quick visualization to PNG
    # Create an 8-bit visualization by clipping to percentile range and normalizing
    vis = np.zeros_like(Lr, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.nanpercentile(depth[valid], [5, 95])
        depth_clip = np.clip(depth, lo, hi)
        depth_clip = np.nan_to_num(depth_clip, nan=0.0)
        vis = cv2.normalize(depth_clip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Write the visualization to disk
    cv2.imwrite(out_path, vis)

if __name__ == '__main__':
    # Path to the trained YOLO weights; expects a .pt file
    model_path = r"\\Truenas\hub-data\FOR2591\jazmin\runs_yolo\selected_models_yolo\mouse-top\runs-all-data\detection-am32gkmp\weights\best.pt"
    # Left and right image paths for stereo pair
    left_img  = r"\\Truenas\hub-data\FOR2591\jazmin\3D_reconstruction\matched_frames\topcam_000003_2023-12-13_11-30-40_2023_12_13T_11_00_11_Picam_frameH264_23098.png"
    right_img = r"\\Truenas\hub-data\FOR2591\jazmin\3D_reconstruction\matched_frames\topcam_000003_2023-12-13_11-30-40_2023_12_13T_11_00_11_Picam_frameMP4_0.png"

    # Known physical baseline between cameras in meters (24 mm)
    baseline_m = 0.024  # 24 mm

    # sensor/native resolution
    # Native sensor width/height used to derive intrinsics at full resolution
    sensor_resolution_width  = 3280   # W0
    sensor_resolution_height = 2464   # H0

    # build native intrinsics from physical specs at native sensor size
    # Focal length and pixel size in millimeters; convert to pixels at native size
    f_mm = 3.04             # mm
    pixel_size_mm = 0.00112 # mm (1.12 µm)
    fx0 = f_mm / pixel_size_mm   # in pixels at native width
    fy0 = f_mm / pixel_size_mm   # assume square pixels at native
    # Principal point assumed at the center of the native sensor
    cx0 = sensor_resolution_width  / 2.0
    cy0 = sensor_resolution_height / 2.0

    # read actual image sizes, then build a K per image
    # Load the actual stereo images in grayscale for subsequent processing
    imgL = cv2.imread(left_img,  cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(right_img, cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        # Immediate, explicit failure if images can't be found/read
        raise FileNotFoundError("Could not read left or right image. Check the paths, not your faith in math.")

    # Extract image dimensions; OpenCV returns shape as (H, W)
    WL, HL = imgL.shape[1], imgL.shape[0]  # left width, height (e.g., 1640 x 928)
    WR, HR = imgR.shape[1], imgR.shape[0]  # right width, height (e.g., 1632 x 918)

    # Compute per-image intrinsics by scaling native intrinsics to actual sizes
    K_left  = make_K_from_native(sensor_resolution_width, sensor_resolution_height, fx0, fy0, cx0, cy0, WL, HL)
    K_right = make_K_from_native(sensor_resolution_width, sensor_resolution_height, fx0, fy0, cx0, cy0, WR, HR)


    # build per-image projection matrices
    # Assume rectified geometry for triangulation: identity rotation and translation along X
    R = np.eye(3)
    T = np.array([[baseline_m], [0.0], [0.0]], dtype=np.float64)
    
    print("Computing dense depth…")
    # Compute and save a quick depth visualization using SGBM on rectified pair
    compute_depth_png(imgL, imgR, K_left, K_right, R, T, baseline_m, out_path="depth_map.png")

    # Construct 3x4 projection matrices for each camera: [K | K*[R|t]]
    P1 = K_left  @ np.hstack((np.eye(3), np.zeros((3, 1), dtype=np.float64)))
    P2 = K_right @ np.hstack((R,T))
    
    

    # Initialize YOLO model from provided weights
    model = YOLO(model_path)

    # YOLO detection
    # Run object detection on both images and print structured results
    det_left = detect_features(model, left_img)
    print(f"detections_left:{det_left}")
    det_right = detect_features(model, right_img)
    print(f"detections_right:{det_right}")

    # SIFT matches inside YOLO regions for class 'Body'
    # Currently uses full-image SIFT matching; later can be restricted to ROIs
    sift_matches = sift_matches_full_image(left_img, right_img)
    print(f"SIFT matches in YOLO 'body' regions: {len(sift_matches)}")

    # 3D reconstruction from SIFT matches
    # Triangulate matched points using the two projection matrices
    pts3d_triang = triangulate_with_proj(sift_matches, P1, P2)
    for x, y, z, cls in pts3d_triang:
        # Print each triangulated point with its label and coordinates in meters
        print(f'{cls}: X={x:.4f} m, Y={y:.4f} m, Z={z:.4f} m')

    # Save mesh
    # Derive an output name from the left image filename to keep pairs together
    name = left_img.split('topcam_')[1].split('.png')[0]
    
    # Reconstruct a surface from the point cloud and write it as a .glb file
    save_mouse_mesh(
        pts3d_triang,
        fr"\\Truenas\hub-data\FOR2591\jazmin\3D_reconstruction\3d_Mouse\mouse_all_{name}.glb"
    )
