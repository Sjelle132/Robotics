import argparse
import math
import os
import sys
import inspect
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Depth from two photos taken by the same camera at two robot poses.
# The rectification and disparity steps follow finalProject_CalibrationPart.ipynb
# (PFAS). There the relative pose R, T came from stereoCalibrate. Here it comes
# from the forward kinematics of the two joint configurations, so T is in mm.
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, os.path.join(currentdir, "11B"))
from kinematics import FK  # noqa: E402  (11B/kinematics.py, angles in radians)

CALIBRATION_FILE = Path(currentdir) / "camera_calibration.npz"
DEPTH_DIR = Path(currentdir) / "DepthPictures"
OUTPUT_DIR = Path(currentdir) / "depth_result"

# Joint angles J1..J4 in degrees for the two photos, same convention as
# MyRobot.move_j. The photos in DepthPictures fit a 5 degree change in J1 with
# the camera looking straight down (q2 + q3 + q4 = -90). The first photo by
# file name was taken at the larger J1, so pose 2 is J1 - 5 here. Only the
# difference matters, but the sign must match the photo order.
DEPTH_POSE_1 = (-12.02, -7.99, -95.76, 13.75)
DEPTH_POSE_2 = (DEPTH_POSE_1[0] - 5.0, *DEPTH_POSE_1[1:])

# Camera orientation in frame 5 (columns are the OpenCV camera axes x, y, z
# written in frame 5). Assumes the camera looks straight down when link 4 is
# horizontal (z_c = -y5) and the top of the image points away from the robot
# (y_c = -x5). Change this if your camera is mounted differently.
R_5C = np.column_stack(([0, 0, 1], [-1, 0, 0], [0, -1, 0])).astype(float)

# Height of the table in the base frame, used only for the expected depth check.
TABLE_Z_MM = 0.0


def read_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def camera_pose(joints_deg):
    """4x4 pose of the camera (OpenCV axes) in the base frame, mm."""
    T05 = FK(np.radians(joints_deg))[5]
    T0C = np.eye(4)
    T0C[:3, :3] = T05[:3, :3] @ R_5C
    T0C[:3, 3] = T05[:3, 3]
    return T0C


def relative_pose(pose_1, pose_2):
    """R, T mapping camera-1 coordinates to camera-2 coordinates (as stereoCalibrate)."""
    T21 = np.linalg.inv(camera_pose(pose_2)) @ camera_pose(pose_1)
    return T21[:3, :3], T21[:3, 3:].copy()


def rectify(mtx, dist, image_size, pose_1, pose_2):
    R, T = relative_pose(pose_1, pose_2)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        mtx, dist, mtx, dist, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    return R, T, R1, R2, P1, P2, Q


def row_difference(gray_1, gray_2, mtx, dist, R1, P1, R2, P2):
    """Rectified row difference of SIFT matches; close to zero when R, T are right."""
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_1, None)
    kp2, des2 = sift.detectAndCompute(gray_2, None)
    if des1 is None or des2 is None:
        return None
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 5:
        return None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    rect1 = cv2.undistortPoints(pts1, mtx, dist, R=R1, P=P1).reshape(-1, 2)
    rect2 = cv2.undistortPoints(pts2, mtx, dist, R=R2, P=P2).reshape(-1, 2)
    return np.abs(rect1[:, 1] - rect2[:, 1]), rect1[:, 0] - rect2[:, 0]


def compute_depth(image_1, image_2, pose_1=DEPTH_POSE_1, pose_2=DEPTH_POSE_2,
                  calibration=CALIBRATION_FILE, output_dir=OUTPUT_DIR, show=True):
    calib = np.load(calibration)
    mtx, dist = calib["mtx"], calib["dist"]
    image_size = tuple(int(v) for v in calib["image_size"])

    first, second = read_image(image_1), read_image(image_2)
    if first is None or second is None:
        raise RuntimeError("Could not read both depth photos.")
    if first.shape[:2] != second.shape[:2] or first.shape[1::-1] != image_size:
        raise RuntimeError(f"Use photos at the calibration resolution {image_size}.")

    R, T, R1, R2, P1, P2, Q = rectify(mtx, dist, image_size, pose_1, pose_2)
    if abs(P2[1, 3]) > abs(P2[0, 3]):
        raise RuntimeError("The two poses give a vertical baseline. Move J1 (sideways) between the photos.")
    if P2[0, 3] > 0:
        # Camera 2 is on the left. SGBM needs the left image first, so swap.
        first, second, pose_1, pose_2 = second, first, pose_2, pose_1
        R, T, R1, R2, P1, P2, Q = rectify(mtx, dist, image_size, pose_1, pose_2)

    baseline = float(np.linalg.norm(T))
    angle = np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))
    cam_height = camera_pose(pose_1)[2, 3] - TABLE_Z_MM
    focal = P1[0, 0]
    print(f"Baseline from FK: {baseline:.1f} mm, relative rotation {angle:.1f} deg")
    print(f"Camera height above the table from FK: {cam_height:.1f} mm")

    # Rectify the full images so pixel coordinates still match P1, P2 and Q.
    map_1x, map_1y = cv2.initUndistortRectifyMap(mtx, dist, R1, P1, image_size, cv2.CV_32FC1)
    map_2x, map_2y = cv2.initUndistortRectifyMap(mtx, dist, R2, P2, image_size, cv2.CV_32FC1)
    rectified_1 = cv2.remap(first, map_1x, map_1y, cv2.INTER_LINEAR)
    rectified_2 = cv2.remap(second, map_2x, map_2y, cv2.INTER_LINEAR)
    gray_1 = cv2.cvtColor(rectified_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(rectified_2, cv2.COLOR_BGR2GRAY)

    check = row_difference(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), mtx, dist, R1, P1, R2, P2)
    if check is not None:
        row_error, match_disparity = check
        print(f"Rectified row difference of {len(row_error)} feature matches: "
              f"median {np.median(row_error):.2f} px, 95th percentile {np.percentile(row_error, 95):.2f} px")
        print(f"Feature disparity: median {np.median(match_disparity):.1f} px "
              f"-> depth {focal * baseline / np.median(match_disparity):.1f} mm")
        # A few pixels are expected from servo play and the camera mount.
        if np.median(row_error) > 5:
            print("WARNING: matching points are not on the same rows. The poses or R_5C "
                  "do not match the photos, so the disparity map will be wrong.")

    # Disparity search range from the expected depth (OpenCV needs a multiple of 16).
    expected_disparity = focal * baseline / max(cam_height, 1.0)
    min_disparity = 0
    num_disparities = max(64, 16 * math.ceil(1.5 * expected_disparity / 16))
    block_size = 5
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size**2,
        P2=32 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
    )
    # SGBM returns disparity in sixteenths of a pixel.
    disparity = stereo.compute(gray_1, gray_2).astype(np.float32) / 16
    disparity[disparity <= min_disparity] = np.nan

    # Depth along the optical axis of the rectified first camera, in mm.
    points = cv2.reprojectImageTo3D(np.nan_to_num(disparity, nan=-1), Q)
    depth = np.abs(points[:, :, 2])
    depth[np.isnan(disparity)] = np.nan
    valid = np.isfinite(depth)
    if valid.any():
        print(f"SGBM depth: median {np.nanmedian(depth):.1f} mm over {valid.mean() * 100:.0f}% of the pixels "
              f"(expected disparity {expected_disparity:.0f} px, search range {num_disparities} px)")
    else:
        print("SGBM found no valid disparities.")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "depth.npz", disparity=disparity, depth=depth, Q=Q, R=R, T=T,
             P1=P1, P2=P2, R1=R1, R2=R2, pose_1=pose_1, pose_2=pose_2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axes[0, 0].imshow(cv2.cvtColor(rectified_1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Rectified photo 1")
    axes[0, 1].imshow(cv2.cvtColor(rectified_2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Rectified photo 2")
    for ax in axes[0]:
        for row in range(60, image_size[1], 60):
            ax.axhline(row, color="lime", linewidth=0.6)
    cmap = plt.get_cmap("viridis").with_extremes(bad="black")
    shown = axes[1, 0].imshow(disparity, cmap=cmap, vmin=min_disparity,
                              vmax=min_disparity + num_disparities - 1)
    axes[1, 0].set_title("Disparity")
    fig.colorbar(shown, ax=axes[1, 0], label="Disparity (pixels)")
    # A few tiny disparities give huge depths, so clip the colour range.
    low, high = np.nanpercentile(depth, (2, 98)) if valid.any() else (0, 1)
    shown = axes[1, 1].imshow(depth, cmap=cmap, vmin=low, vmax=high)
    axes[1, 1].set_title("Depth")
    fig.colorbar(shown, ax=axes[1, 1], label="Depth (mm)")
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(output_dir / "depth.png", dpi=120)
    print(f"Saved {output_dir / 'depth.npz'} and {output_dir / 'depth.png'}")
    if show:
        plt.show()
    return disparity, depth


if __name__ == "__main__":
    default_photos = sorted(DEPTH_DIR.glob("*.png"))
    parser = argparse.ArgumentParser(description="Disparity and depth from two photos at two robot poses.")
    parser.add_argument("photos", nargs="*", type=Path, default=default_photos[:2],
                        help="Photo from pose 1 and photo from pose 2 (default: the two in DepthPictures).")
    parser.add_argument("--pose1", type=float, nargs=4, default=DEPTH_POSE_1, metavar="DEG",
                        help="J1 J2 J3 J4 in degrees for photo 1.")
    parser.add_argument("--pose2", type=float, nargs=4, default=DEPTH_POSE_2, metavar="DEG",
                        help="J1 J2 J3 J4 in degrees for photo 2.")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_FILE,
                        help="Output of MyRobot_calibration.py (default: camera_calibration.npz).")
    parser.add_argument("--no-show", action="store_true", help="Only save the figure.")
    args = parser.parse_args()
    if len(args.photos) != 2:
        parser.exit(1, "Error: give exactly two photos.\n")
    try:
        compute_depth(*args.photos, tuple(args.pose1), tuple(args.pose2),
                      args.calibration, show=not args.no_show)
    except (OSError, RuntimeError, KeyError, cv2.error) as error:
        parser.exit(1, f"Error: {error}\n")
