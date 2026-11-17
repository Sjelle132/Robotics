import argparse
import os
import sys
import inspect
import math
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# --- Dynamixel SDK path bootstrap -----------------------
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sdk_path = os.path.join(parentdir, "DynamixelSDK", "python", "src")
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

from dynamixel_sdk import PortHandler, PacketHandler

# Photo test settings. Angles are J1, J2, J3, J4 in degrees after applying the
# controller's zero offsets. The example keeps the last link horizontal in
# the nominal model. Check the camera faces straight down on your actual arm.
CAMERA_ID = 0
PHOTO_POSE_1 = (-12.02, -7.99, -95.76, 13.75)
PHOTO_POSE_2 = (PHOTO_POSE_1[0] + 10.0, *PHOTO_POSE_1[1:])


def cosd(x_deg):  # cos with degrees
    return math.cos(math.radians(x_deg))


def sind(x_deg):  # sin with degrees
    return math.sin(math.radians(x_deg))


class MyRobot:
    # ----------------------------- Constants (AX-12A) -------------------------
    ADDR_MX_TORQUE_ENABLE = 24       # 1 byte
    ADDR_MX_GOAL_POSITION = 30       # 2 bytes
    ADDR_MX_MOVING_SPEED = 32        # 2 bytes
    ADDR_MX_TORQUE_LIMIT = 34        # 2 bytes
    ADDR_MX_PRESENT_POSITION = 36    # 2 bytes

    PROTOCOL_VERSION = 1.0
    BAUDRATE = 1000000
    DEVICENAME = "COM5"              # e.g., Windows: 'COM3' | Linux: '/dev/ttyUSB0' | Mac: '/dev/tty.usbserial-*'

    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0
    COMM_SUCCESS = 0
    DXL_MOVING_STATUS_THRESHOLD = 10  # [ticks], not used directly here
    TICKS_PER_DEG = 1.0 / 0.29        # ≈ 3.448275862 ticks per degree
    DEG_PER_TICK = 0.29

    # ------------------------------- Init -------------------------------------
    def __init__(self, start_pose=(0, 0, 0, 0)):
        # Robot + kinematics state
        self.motor_ids = [1, 2, 3, 4]
        self.gripper_motor_id = 5
        self.forward_transform = np.zeros((4, 4))
        self.joint_angles = np.array([0.0, 0.0, 0.0, 0.0])  # deg
        self.joint_pos = np.zeros((4, 4))
        self.draw_robot_flag = False
        self.use_smooth_speed_flag = False
        self.gripper_open_flag = True

        # deg limits: [-130,130], [-180,0], [-140,140], [-140,100]
        self.joint_limits = np.array([[-90, 90], [-180, 180], [-140, 140], [-140, 140]])

        # IK config (simple analytical below; no external toolbox)
        self.ik_weights = np.array([0.25, 0.25, 0.25, 1, 1, 1])

        # Motor offsets (deg)
        self.joint_offsets = np.array([150, 150, 150, 150], dtype=float)

        self.joint_angle_error = np.array([0.0, 0.0, 0.0, 0.0])
        self.init_status = 0

        self.movement_history = []  # list of dicts
        self.motor_speed = np.array([0.1, 0.1, 0.1, 0.1])   # [0..1]
        self.motor_torque = np.array([1.0, 1.0, 1.0, 1.0])  # [0..1]
        self.pitch = 0.0

        # --- SDK setup ---
        try:
            self.port_handler = PortHandler(self.DEVICENAME)
            self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)

            if self.port_handler.openPort():
                print("Succeeded to open the port!")
            else:
                print("Failed to open the port! Reconnect Robot.")
                raise RuntimeError("openPort failed")

            if self.port_handler.setBaudRate(self.BAUDRATE):
                print("Succeeded to change the baudrate!")
            else:
                print("Failed to change the baudrate! Reconnect Robot.")
                raise RuntimeError("setBaudRate failed")

            # Initialize defaults
            self.set_speed(self.motor_speed, overwrite_speeds=True)
            self.set_torque_limit(self.motor_torque)
            # Set the requested goal before enabling torque, then wait for arrival.
            self.move_j(*start_pose, wait=False)
            self.enable_motors()
            self._wait_for_joints(self.joint_angles, tol_deg=2.0, poll_period=0.05,
                                  timeout_s=15.0)
            self.init_status = 1

        except Exception as e:
            print(f"Initialization error: {e}")
            if hasattr(self, "port_handler") and self.port_handler.is_open:
                self.port_handler.closePort()

    # ----------------------------- Utilities ----------------------------------
    @staticmethod
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    @classmethod
    def deg_to_ticks(cls, deg):
        return int(round(deg * cls.TICKS_PER_DEG))

    @classmethod
    def ticks_to_deg(cls, ticks):
        return ticks * cls.DEG_PER_TICK

    def _check_limits_one(self, deg, idx):
        lo, hi = self.joint_limits[idx]
        if not (lo <= deg <= hi):
            raise ValueError(f"Joint {idx + 1} target {deg:g} is outside [{lo}, {hi}] degrees.")
        return deg

    # --------------------------- Motor I/O wrappers ---------------------------
    def _write1(self, dxl_id, addr, value):
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, addr, int(value))
        if dxl_comm_result != self.COMM_SUCCESS or dxl_error != 0:
            raise IOError(f"DXL write1 error id={dxl_id}, addr={addr}, result={dxl_comm_result}, error={dxl_error}")

    def _write2(self, dxl_id, addr, value):
        dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(self.port_handler, dxl_id, addr, int(value))
        if dxl_comm_result != self.COMM_SUCCESS or dxl_error != 0:
            raise IOError(f"DXL write2 error id={dxl_id}, addr={addr}, result={dxl_comm_result}, error={dxl_error}")

    def _read2(self, dxl_id, addr):
        value, dxl_comm_result, dxl_error = self.packet_handler.read2ByteTxRx(self.port_handler, dxl_id, addr)
        if dxl_comm_result != self.COMM_SUCCESS or dxl_error != 0:
            raise IOError(f"DXL read2 error id={dxl_id}, addr={addr}, result={dxl_comm_result}, error={dxl_error}")
        return value

    # ------------------------------- Speeds/Torque ----------------------------
    def smooth_speed(self, joint_angles_delta_deg):
        max_angle = float(np.max(np.abs(joint_angles_delta_deg)))
        if max_angle == 0:
            return
        speed_per_deg = max_angle / 100.0
        if speed_per_deg != 0:
            new_speeds = np.abs(joint_angles_delta_deg / speed_per_deg) * 0.01
            # preserve zeros by falling back to previous speeds
            for i in range(len(self.motor_speed)):
                if new_speeds[i] == 0:
                    new_speeds[i] = self.motor_speed[i]
                else:
                    new_speeds[i] = new_speeds[i] * self.motor_speed[i]
            self.set_speed(new_speeds, overwrite_speeds=False)

    def set_speed(self, speeds, overwrite_speeds=True):
        speeds = np.asarray(speeds, dtype=float)
        if overwrite_speeds:
            self.motor_speed = speeds.copy()

        for i, dxl_id in enumerate(self.motor_ids):
            s = float(speeds[i])
            if 0 < s <= 1:
                raw = int(round(s * 1023))
                self._write2(dxl_id, self.ADDR_MX_MOVING_SPEED, raw)
            else:
                print("Movement speed out of range, enter value between (0, 1].")

    def set_torque_limit(self, torques):
        torques = np.asarray(torques, dtype=float)
        self.motor_torque = torques.copy()
        for i, dxl_id in enumerate(self.motor_ids):
            t = float(torques[i])
            if 0 < t <= 1:
                self._write2(dxl_id, self.ADDR_MX_TORQUE_LIMIT, int(round(t * 1023)))
            else:
                print("Torque limit out of range, enter value between (0, 1].")

    # --------------------------------- Power ----------------------------------
    def enable_motors(self):
        for dxl_id in self.motor_ids:
            self._write1(dxl_id, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_ENABLE)
        print("Dynamixel torque enabled on all motors.")

    def disable_motors(self):
        for dxl_id in self.motor_ids:
            try:
                self._write1(dxl_id, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_DISABLE)
            except Exception as e:
                print(f"Disable motor {dxl_id} error: {e}")
        try:
            self.port_handler.closePort()
        finally:
            self.init_status = 0
        print("Dynamixels disabled and port closed.")

    # ------------------------------- Telemetry --------------------------------
    def get_position(self, motor_id):
        ticks = self._read2(motor_id, self.ADDR_MX_PRESENT_POSITION)
        return self.ticks_to_deg(ticks)

    def read_joint_angles(self):
        j_a = np.zeros(4, dtype=float)
        for i, dxl_id in enumerate(self.motor_ids):
            ticks = self._read2(dxl_id, self.ADDR_MX_PRESENT_POSITION)
            deg = self.ticks_to_deg(ticks) - self.joint_offsets[i]
            j_a[i] = deg
            self.joint_angle_error[i] = deg - self.joint_angles[i]
        return j_a

    # --------------------------- Kinematics (FK / IK) -------------------------
   # Please input your forward and inverse kinematics here for position and pose conversion to joint angles.
   #  The output should be in the format of an array

    # ------------------------------- Motion -----------------------------------
    def move_j(self, j1_deg, j2_deg, j3_deg, j4_deg, wait=True, tol_deg=2.0,
               poll_period=0.05, timeout_s=15.0):
        # Limits
        desired = np.array([j1_deg, j2_deg, j3_deg, j4_deg], dtype=float)
        for i in range(4):
            desired[i] = self._check_limits_one(desired[i], i)

        # Smooth speed option
        if self.use_smooth_speed_flag:
            self.smooth_speed(desired - self.joint_angles)

        # Apply motor offsets for command
        cmd_deg = desired + self.joint_offsets
        ticks = [self.deg_to_ticks(deg) for deg in cmd_deg]
        if not all(0 <= tick <= 1023 for tick in ticks):
            raise ValueError("Joint target is outside the AX-12A position range.")
        for dxl_id, tick in zip(self.motor_ids, ticks):
            self._write2(dxl_id, self.ADDR_MX_GOAL_POSITION, tick)
        self.joint_angles = desired
        if wait:
            self._wait_for_joints(desired, tol_deg, poll_period, timeout_s)

    def _wait_for_joints(self, desired, tol_deg, poll_period, timeout_s):
        deadline = time.monotonic() + timeout_s
        measured = None
        read_error = None
        while time.monotonic() < deadline:
            try:
                measured = self.read_joint_angles()
                read_error = None
                if np.all(np.abs(measured - desired) <= tol_deg):
                    return
            except OSError as error:
                # A transient Dynamixel read failure can recover on the next poll.
                read_error = str(error)
            time.sleep(poll_period)
        details = [f"Target {desired.round(2)} degrees, tolerance {tol_deg:g} degrees."]
        if measured is not None:
            details.append(f"Last measured {measured.round(2)} degrees.")
        if read_error is not None:
            details.append(f"Last read error: {read_error}")
        raise TimeoutError(f"Move timed out after {timeout_s:g} s. " + " ".join(details))

    #def move_c(self, x, y, z, pitch_deg, **kwargs):
     #   j_deg = self.inverse(x, y, z, math.radians(pitch_deg))
      #  self.move_j(j_deg[0], j_deg[1], j_deg[2], j_deg[3], **kwargs)

    #def read_ee_position(self):
     #   j_a = self.read_joint_angles()
      #  return self.forward(j_a)

    # --------------------------- History / Macros ------------------------------
    def record_configuration(self):
        entry = {
            "j1": float(self.joint_angles[0]),
            "j2": float(self.joint_angles[1]),
            "j3": float(self.joint_angles[2]),
            "j4": float(self.joint_angles[3]),
            "speed": float(self.motor_speed[0]),
            "torque": float(self.motor_torque[0]),
            "gripper_open": bool(self.gripper_open_flag),
        }
        self.movement_history.append(entry)
        print(
            f"Recorded Speed: {entry['speed']:.3f}, Torque: {entry['torque']:.3f}\n"
            f"Joint Positions: {entry['j1']:.2f}, {entry['j2']:.2f}, {entry['j3']:.2f}, {entry['j4']:.2f}\n"
            f"Gripper open: {int(entry['gripper_open'])}"
        )

    def delete_last_recorded_configuration(self):
        if not self.movement_history:
            print("No last history position")
            return
        self.movement_history.pop()

    def play_configuration_history(self, dwell_after_move_s=1.0, dwell_after_grip_s=3.0):
        if not self.movement_history:
            print("No recorded configurations.")
            return
        for rec in self.movement_history:
            s = rec["speed"]
            t = rec["torque"]
            self.set_speed([s, s, s, s], overwrite_speeds=True)
            self.set_torque_limit([t, t, t, t])
            self.move_j(rec["j1"], rec["j2"], rec["j3"], rec["j4"])
            time.sleep(dwell_after_move_s)
            if self.gripper_open_flag != rec["gripper_open"]:
                self.actuate_gripper()
                time.sleep(dwell_after_grip_s)


def save_photo(path, frame):
    import cv2

    encoded_ok, encoded = cv2.imencode(".png", frame)
    if not encoded_ok:
        raise RuntimeError(f"Could not encode {path.name}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def capture_calibration_photos(camera_id=CAMERA_ID):
    """Collect ten checkerboard photos while the operator repositions the board."""
    import cv2

    count = 10
    window = "Checkerboard calibration"
    folder = Path(currentdir) / "calibration_photos" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    camera = cv2.VideoCapture(camera_id, backend)
    saved = 0
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}. Check --camera.")

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while saved < count:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read a frame from camera {camera_id}.")

            preview = frame.copy()
            label = f"{saved}/{count} saved | Any key: capture | Q/Esc: quit"
            cv2.putText(preview, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window, preview)
            key = cv2.waitKeyEx(30)
            try:
                visible = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                break  # The user closed the window.
            if visible < 1 or key in (ord("q"), ord("Q"), 27):
                break
            if key == -1:
                continue

            path = folder / f"checkerboard_{saved + 1:02d}.png"
            save_photo(path, frame)
            saved += 1
            print(f"Saved {saved}/{count}  {path}", flush=True)
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Finished with {saved}/{count} photos.")


# ------------------------------- Two-photo test -------------------------------
def capture_frame(camera, settle_s=0.6):
    """Read through the settling period to discard buffered frames from motion."""
    start = time.monotonic()
    while time.monotonic() < start + settle_s + 2.0:
        ok, frame = camera.read()
        if ok and frame is not None and time.monotonic() >= start + settle_s:
            return frame.copy()
        if not ok or frame is None:
            time.sleep(0.05)
    raise RuntimeError("Camera did not return an image.")


def take_two_photos(camera_id=CAMERA_ID):
    import cv2

    window = "Robot photos (pose 1 on the left, pose 2 on the right)"
    folder = Path(currentdir) / "photos" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    camera = None
    robot = None
    frames = []
    preview = True
    window_opened = False
    success = False
    try:
        print(f"Opening camera {camera_id}.", flush=True)
        camera = cv2.VideoCapture(camera_id, backend)
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}. Check --camera.")

        for number, pose in enumerate((PHOTO_POSE_1, PHOTO_POSE_2), start=1):
            print(f"Moving to pose {number} on {MyRobot.DEVICENAME}: {pose} degrees.", flush=True)
            if number == 1:
                robot = MyRobot(start_pose=pose)
                if not robot.init_status:
                    raise RuntimeError(f"Could not initialize the robot on {MyRobot.DEVICENAME}.")
            else:
                robot.move_j(*pose)

            print(f"Pose {number} reached. Capturing photo {number}.", flush=True)
            frame = capture_frame(camera)
            path = folder / f"photo_{number}.png"
            save_photo(path, frame)
            print(f"Saved {path}", flush=True)
            frames.append(frame)

            if preview:
                try:
                    if not window_opened:
                        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                        window_opened = True
                    cv2.imshow(window, np.hstack(frames))
                    cv2.waitKey(1)
                except cv2.error as error:
                    print(f"Preview unavailable. Open the saved PNGs to view the photos. {error}",
                          file=sys.stderr)
                    preview = False
        success = True
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
    finally:
        if camera is not None:
            camera.release()
        if robot is not None and robot.init_status:
            # Close communications while leaving the arm holding the final pose.
            robot.port_handler.closePort()

    if window_opened:
        try:
            if preview:
                print("Close the photo window or press Q or Esc in it to finish.", flush=True)
                while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:
                    if cv2.waitKey(50) & 0xFF in (ord("q"), ord("Q"), 27):
                        break
        except cv2.error:
            # Some OpenCV backends raise when the window has already closed.
            pass
        finally:
            cv2.destroyAllWindows()
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the two-pose photo test or collect checkerboard photos.")
    parser.add_argument("--calibration", action="store_true", help="Capture ten checkerboard photos.")
    parser.add_argument("--camera", type=int, default=CAMERA_ID,
                        help=f"Camera index (default: {CAMERA_ID}).")
    args = parser.parse_args()
    if args.calibration:
        import cv2

        try:
            capture_calibration_photos(args.camera)
        except (OSError, RuntimeError, cv2.error) as error:
            parser.exit(1, f"Error: {error}\n")
    else:
        sys.exit(0 if take_two_photos(args.camera) else 1)
#Example 2:
#Demonstrates how to move the robot joints into desired a list of joint positions [degrees]
#You might need to adjust the offsets in the code above to make the zero point suit your needs.
#Just comment out the code snippet, input desired joint angle sequence in robot.move_j() and run.

'''
if __name__ == "__main__":
    import time
    import numpy as np

    # Create robot instance
    robot = MyRobot()

    if robot.init_status:
        try:
            # ----------------- Define the sequence of joint angles -----------------
            # Each row = [J1, J2, J3, J4] in DEGREES
            # Adjust these based on your mechanical limits / workspace
            joint_sequence = np.array([
                [-12.02, -7.99, -95.76, 13.75],
                [-11.75, -7.21, -93.14, 10.35],
                [-11.24, -6.60, -90.47, 7.07],
                [-10.49, -6.14, -87.85, 3.99],
                [-9.33, -5.47, -85.76, 1.24],
                [-7.69, -5.20, -83.63, -1.17],
                [-5.96, -5.05, -81.82, -3.13],
                [-4.25, -4.67, -80.78, -4.55],
                [-2.04, -4.61, -79.95, -5.44],
                [-0.00, -4.60, -79.66, -5.74],
                [2.04, -4.61, -79.95, -5.44],
                [4.25, -4.67, -80.78, -4.55],
                [5.96, -5.05, -81.82, -3.13],
                [7.69, -5.20, -83.63, -1.17],
                [9.33, -5.47, -85.76, 1.24],
                [10.49, -6.14, -87.85, 3.99],
                [11.24, -6.60, -90.47, 7.07],
                [11.75, -7.21, -93.14, 10.35],
                [12.02, -7.99, -95.76, 13.75],
                [11.75, -8.85, -98.32, 17.17],
                [11.24, -9.82, -100.68, 20.50],
                [10.49, -10.87, -102.80, 23.67],
                [9.33, -11.60, -105.02, 26.62],
                [7.69, -12.54, -106.61, 29.14],
                [5.96, -13.37, -107.84, 31.21],
                [4.25, -13.76, -109.06, 32.82],
                [2.04, -14.16, -109.60, 33.76],
                [-0.00, -14.30, -109.78, 34.08],
                [-2.04, -14.16, -109.60, 33.76],
                [-4.25, -13.76, -109.06, 32.82],
                [-5.96, -13.37, -107.84, 31.21],
                [-7.69, -12.54, -106.61, 29.14],
                [-9.33, -11.60, -105.02, 26.62],
                [-10.49, -10.87, -102.80, 23.67],
                [-11.24, -9.82, -100.68, 20.50],
                [-11.75, -8.85, -98.32, 17.17],
                [-12.02, -7.99, -95.76, 13.75]
                # return home
            ])

            # ----------------- Run the motion sequence -----------------
            robot.enable_motors()
            print("\nStarting joint sequence...")
            time.sleep(1.0)

            #Number of loops
            loops = 5

            for loop in range(loops):
                print(f"\n🔁 Loop {loop + 1} of {loops} ----------------------------")
                for idx, pose in enumerate(joint_sequence):
                    print(f"\n➡️  Moving to pose {idx+1}: {pose}°")
                    robot.move_j(*pose)
                    robot.record_configuration()  # save to history
                    # Read and print measured joint positions
                    measured = robot.read_joint_angles()
                    print("Measured (deg):", np.round(measured, 2))
                    time.sleep(0)  # pause between poses



        except Exception as e:
            print("Error during sequence:", e)

        finally:
            robot.disable_motors()
            print("\nMotors disabled. Program finished.")
'''
