import numpy as np
import cv2 as cv
import glob

# Termination criteria for cornerSubPix
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points for an 8x11 checkerboard (10x7 inner corners)
objp = np.zeros((7*10, 3), np.float32)
objp[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load images from the specified folder
images = glob.glob('un_dist/*.jpg')

if not images:
    print("No images found. Check the path and file pattern.")
else:
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Optional: Apply Gaussian blur to the image
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, (10, 7), None)

        # If found, add object points and image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (10, 7), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
        else:
            print(f"Checkerboard corners not found in image: {fname}")

    cv.destroyAllWindows()

    # Check if any points were added before performing calibration
    if objpoints and imgpoints:
        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        if ret:
            # Save the calibration results
            np.savez('iphone_matrix.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

            # Print calibration results
            print(f"Camera matrix:\n{mtx}")
            print(f"Distortion coefficients:\n{dist}")
        else:
            print("Calibration failed. Check if sufficient images with detectable checkerboard patterns were provided.")
    else:
        print("No valid points were found. Ensure your images contain a detectable checkerboard pattern.")


