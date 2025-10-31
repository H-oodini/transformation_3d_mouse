import os
import numpy as np
import cv2 as cv
import glob
import subprocess
        
def setup_camera(image_glob="*.jpg", squares=(7, 8), square_size=30.0, show=False):
    """
    Calibrate camera from chessboard images.

    Parameters
    ----------
    image_glob : str
        Glob for calibration images.
    squares : (int, int)
        Number of squares on the board (cols, rows). For 7x8 squares, use (7, 8).
    square_size : float
        Size of one square in real-world units (e.g., millimeters). Used to scale object points.
    show : bool
        If True, draw detected corners and briefly display each image.

    Returns
    -------
    retval : float
        RMS reprojection error from cv.calibrateCamera.
    cameraMatrix : np.ndarray
    distCoeffs : np.ndarray
    rvecs : list
    tvecs : list
    """

    # Inner corners = squares - 1 in each dimension
    cols, rows = squares[0] - 1, squares[1] - 1
    pattern_size = (cols, rows)  # (columns, rows) of inner corners

    # Termination criteria for cornerSubPix
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    # Prepare object points: (0,0,0), (1,0,0), ... scaled by square_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    images = glob.glob(image_glob)
    if not images:
        raise FileNotFoundError(f"No images matched '{image_glob}'")

    for fname in images:
        img = cv.imread(fname)
        if img is None:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Faster, more robust flags; swap to findChessboardCornersSB if available
        flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_FAST_CHECK
        ret, corners = cv.findChessboardCorners(gray, pattern_size, flags)

        if ret:
            # Refine corner locations to sub-pixel
            corners = cv.cornerSubPix(
                gray,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=criteria
            )
            objpoints.append(objp)
            imgpoints.append(corners)

            if show:
                vis = img.copy()
                cv.drawChessboardCorners(vis, pattern_size, corners, ret)
                cv.imshow("chessboard", vis)
                cv.waitKey(150)

    if show:
        cv.destroyAllWindows()

    if not objpoints:
        raise RuntimeError("No chessboard patterns were detected. Check pattern_size or images.")

    # Calibrate
    image_size = (gray.shape[1], gray.shape[0])  # (width, height)
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # Optional: compute mean reprojection error for sanity check
    total_error = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        proj, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
        err = cv.norm(imgpoints[i], proj, cv.NORM_L2)
        total_error += err**2
        total_points += len(proj)
    mean_error = (total_error / total_points) ** 0.5

    
    if mean_error > 1.0:
        print(f"Warning: high reprojection error: {mean_error:.3f} pixels") 
    return retval, cameraMatrix, distCoeffs, rvecs, tvecs, image_size


def main():
  return 0




if __name__ == '__main__':
  main()
