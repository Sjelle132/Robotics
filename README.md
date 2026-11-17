<<<<<<< HEAD
# Robotics 34753 – Problem 11b: probing red smarties

A 4-joint Dynamixel arm (AX-12A servos) with a camera on the end-effector looks at a
cluster of smarties, finds the red ones and probes each with its stylus.

The pipeline:

1. **Camera calibration** – intrinsics and lens distortion from 10 checkerboard photos.
2. **Depth** – two photos from two robot poses, rectified with the relative pose from
   forward kinematics, then a disparity map with StereoSGBM and depth in mm.
3. **Detection** – red smarties found by colour (HSV) in the image.
4. **Inverse kinematics** – pixel → point on the table → joint angles → move the arm.

## Structure

```
MyRobot_beforeCalibration.py   Robot control (Dynamixel SDK), camera capture:
                                 two-pose photos (default) or --calibration photos
MyRobot_calibration.py         Camera calibration from CalibrationPictures/
MyRobot_depth.py               Disparity and depth from the two DepthPictures/
11B/
  kinematics.py                FK, IK and Jacobians (DH model, mm and radians)
  11b.py                       Red-smartie detection and pick pipeline
  trajectory.py                Trajectory planning
  robot.py                     Motor interface stub
  simuler_robot.py             3D simulation
  test_kamera.py               Camera test
  smarties.jpg                 Test image
CalibrationPictures/           10 checkerboard photos (8 x 6 inner corners, 30 mm squares)
DepthPictures/                 Two photos of the smarties, J1 differs by 5 degrees
```

## Requirements

```
pip install numpy opencv-python matplotlib dynamixel-sdk
```

OpenCV 4.4 or newer (SIFT is used for the rectification check). The calibration and
depth scripts do not need the robot connected.

`MyRobot_beforeCalibration.py` adds `../DynamixelSDK/python/src` to the path if the SDK
folder is there; otherwise the pip-installed `dynamixel-sdk` is used. Set `DEVICENAME`
(e.g. `COM5`) to the robot's port.

## Usage

Run from the repository folder.

```
# 1. Calibrate the camera (writes camera_calibration.npz)
python MyRobot_calibration.py            # --show to view an undistorted photo

# 2. Depth from the two photos (writes depth_result/depth.npz and depth.png)
python MyRobot_depth.py                  # --no-show to only save the figure

# New photos: take them with the robot, then pass the photos and the poses
python MyRobot_beforeCalibration.py      # saves photos/<time>/photo_1.png, photo_2.png
python MyRobot_depth.py photos/<time>/photo_1.png photos/<time>/photo_2.png \
    --pose1 -12.02 -7.99 -95.76 13.75 --pose2 -2.02 -7.99 -95.76 13.75

# New checkerboard photos
python MyRobot_beforeCalibration.py --calibration
python MyRobot_calibration.py --folder calibration_photos/<time>
```

## Results with the included photos

| | |
|---|---|
| Calibration reprojection error | 0.42 px (10/10 photos) |
| Focal length | fx 706.6 px, fy 713.5 px |
| Principal point | (370.1, 212.0) px at 640 x 480 |
| Baseline between depth photos (FK) | 12.1 mm (J1 changed 5 degrees) |
| Depth from matched features | 164.5 mm (FK camera height: 165.0 mm) |
| SGBM depth, median | 155 mm, 63 % of pixels |

With a 12 mm baseline one pixel of disparity is about 3 mm of depth, so the 7–8 mm
smarties hardly show in the depth map. A larger J1 change (10–20 degrees) gives better
depth resolution.

## Notes

- `R_5C` in `MyRobot_depth.py` assumes the camera looks straight down when link 4 is
  horizontal, with the top of the image pointing away from the robot.
- The depth script prints the row difference of matched features after rectification.
  More than about 5 px means the poses or the photo order do not match the photos.
- The rectification and disparity code is adapted from
  [PFAS_FinalProject](https://github.com/Sjelle132/PFAS_FinalProject).
=======
# Robotics
>>>>>>> 820c252bd148308dab0f03b36019e2667b7b7bbe
