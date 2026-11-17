import argparse
import os
import sys
import inspect
from pathlib import Path

import cv2
import numpy as np

# Camera calibration from the checkerboard photos.
# Same procedure as finalProject_CalibrationPart.ipynb (PFAS), but for the single
# camera on the robot: find the corners, calibrate, and report the board's
# rotation and translation relative to the camera for each photo.
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

CALIBRATION_DIR = Path(currentdir) / "CalibrationPictures"
OUTPUT_FILE = Path(currentdir) / "camera_calibration.npz"

# The board has 8 x 6 inner corners (9 x 7 squares).
COLS, ROWS = 8, 6
# Measured side length of one square in mm (3 cm). The intrinsics do not depend
# on this value, only the board translations do.
SQUARE_SIZE_MM = 30.0


def read_image(path):
    # np.fromfile + imdecode also works with non-ASCII Windows paths.
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def calibrate(folder=CALIBRATION_DIR, square_size=SQUARE_SIZE_MM, output=OUTPUT_FILE, show=False):
    names = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
    if not names:
        raise RuntimeError(f"No .png or .jpg images in {folder}.")

    pattern_size = (COLS, ROWS)
    objp = np.zeros((COLS * ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[:COLS, :ROWS].T.reshape(-1, 2) * square_size

    objpoints = []
    imgpoints = []
    used_names = []
    # This detector already returns subpixel corner positions.
    corner_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY

    for path in names:
        img = read_image(path)
        if img is None:
            print(f"Skipping {path.name}: could not read it.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=corner_flags)
        if not found:
            print(f"Skipping {path.name}: no {COLS}x{ROWS} checkerboard found.")
            continue
        corners = corners.reshape(-1, 1, 2)
        # A symmetric board can be detected starting from opposite ends. Make the
        # first row run left to right so the board poses are comparable.
        if corners[COLS - 1, 0, 0] < corners[0, 0, 0]:
            corners = corners[::-1].copy()
        objpoints.append(objp)
        imgpoints.append(corners)
        used_names.append(path.name)

    if len(objpoints) < 3:
        raise RuntimeError("Too few images with checkerboard corners.")
    image_size = gray.shape[::-1]
    print(f"Usable images: {len(objpoints)}/{len(names)}")

    # With only ten views the free k3 term overfits (it came out around -24),
    # so it is fixed at zero. The RMS error is almost the same.
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None, flags=cv2.CALIB_FIX_K3
    )

    print(f"\nReprojection RMS: {rms:.2f} px")
    print("Camera matrix K:")
    print(np.array2string(mtx, precision=2, suppress_small=True))
    print("Distortion (k1, k2, p1, p2, k3):", np.round(dist.ravel(), 4))

    # Extrinsics: pose of the board in the camera frame for each photo.
    unit = "mm" if square_size != 1.0 else "squares"
    rotations = []
    print(f"\nBoard pose in the camera frame (translation in {unit}):")
    for name, objp_i, imgp, rvec, tvec in zip(used_names, objpoints, imgpoints, rvecs, tvecs):
        R, _ = cv2.Rodrigues(rvec)
        rotations.append(R)
        projected, _ = cv2.projectPoints(objp_i, rvec, tvec, mtx, dist)
        err = np.sqrt(np.mean(np.sum((projected - imgp) ** 2, axis=2)))
        angle = np.degrees(np.linalg.norm(rvec))
        print(f"  {name[:24]:24s} t = {np.round(tvec.ravel(), 2)}  "
              f"distance {np.linalg.norm(tvec):.2f}  rotation {angle:.1f} deg  error {err:.2f} px")

    np.savez(
        output,
        mtx=mtx, dist=dist, image_size=image_size, rms=rms,
        rvecs=np.array(rvecs), tvecs=np.array(tvecs), rotations=np.array(rotations),
        names=np.array(used_names), square_size=square_size,
    )
    print(f"\nSaved {output}")

    if show:
        img = read_image(folder / used_names[0])
        undistorted = cv2.undistort(img, mtx, dist)
        cv2.namedWindow("Original | undistorted", cv2.WINDOW_NORMAL)
        cv2.imshow("Original | undistorted", np.hstack((img, undistorted)))
        print("Press any key in the image window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return mtx, dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate the robot camera from checkerboard photos.")
    parser.add_argument("--folder", type=Path, default=CALIBRATION_DIR,
                        help="Folder with the checkerboard photos (default: CalibrationPictures).")
    parser.add_argument("--square-size", type=float, default=SQUARE_SIZE_MM,
                        help="Side length of one checkerboard square in mm.")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE,
                        help="Where to save the calibration (default: camera_calibration.npz).")
    parser.add_argument("--show", action="store_true", help="Show one photo before and after undistortion.")
    args = parser.parse_args()
    try:
        calibrate(args.folder, args.square_size, args.output, args.show)
    except (OSError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
