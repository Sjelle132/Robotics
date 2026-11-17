import os
import cv2
import numpy as np
from pathlib import Path

from kinematics import FK, IK
from trajectory import make_traj, sample_traj
from robot import move
from simuler_robot import simuler

os.system('cls')

# Opgave 11b - Røde Smarties

# 1. Indstillinger
folder = Path(__file__).resolve().parent
img_path = folder / 'smarties.jpg'
mode = 'test'         # test / robot
show_img = True
vis_sim = True        # Vis 3D-simulering


# Rød farve i HSV
red1_low  = np.array([0, 130, 90])
red1_high = np.array([6, 255, 255])
red2_low  = np.array([174, 130, 90])
red2_high = np.array([179, 255, 255])

min_area = 1000
min_circ = 0.6
k_open = 3
k_close = 11

# Kameraparametre - foreløbige værdier til simulation
lam = 8.0
alpha_x = 300.0
alpha_y = 300.0
o_r = 250.0
o_c = 250.0

# Test af kameraets placering i robotbasen
# Selvvalgte værdier - ikke fra den fysiske robot

T0C = np.array([
    [1,  0,  0, 150],
    [0, -1,  0,   0],
    [0,  0, -1, 250],
    [0,  0,  0,   1]
], dtype=float)

# Smarties ligger på et vandret plan
z_plane = 100.0  # [mm], selvvalgt testhøjde

# Separat offline-test af IK og baneplanlægning (uden kamera)
test_ik = True
p_test0 = np.array([150., 0., 120.])   # [mm], kun til simulation
p_test1 = np.array([150., 20., 120.])  # [mm], kun til simulation


# 2. Hent billede
def get_image():
    if mode == 'test':
        if not img_path.is_file():
            raise FileNotFoundError(f'Billedet findes ikke: {img_path}')
        data = np.fromfile(str(img_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif mode == 'robot':
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            raise RuntimeError('Kameraet kunne ikke åbnes')
        success, img = cam.read()
        cam.release()
        if not success:
            raise RuntimeError('Kunne ikke tage et billede')
    else:
        raise ValueError('Ukendt tilstand')

    if img is None:
        raise RuntimeError('Kunne ikke indlæse billedet')
    return img


# 3. Find røde pixels
def segment_red(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, red1_low, red1_high)
    mask2 = cv2.inRange(hsv, red2_low, red2_high)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# 4. Find Smarties
def find_smarties(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    marked = img.copy()
    clean_mask = np.zeros_like(mask)

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < min_area or perimeter == 0:
            continue

        circ = 4 * np.pi * area / perimeter**2
        if circ < min_circ:
            continue

        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        c = M['m10'] / M['m00']
        r = M['m01'] / M['m00']
        centers.append((r, c))

        pos = (round(c), round(r))
        cv2.circle(marked, pos, 12, (0, 0, 0), -1)
        cv2.circle(marked, pos, 9, (0, 255, 0), -1)
        cv2.drawContours(clean_mask, [contour], -1, 255, cv2.FILLED)

    return np.array(centers).reshape(-1, 2), marked, clean_mask


# 5. Kamerageometri
def pixel_to_image(r, c):
    u = -(r - o_r) / alpha_x
    v = (c - o_c) / alpha_y
    return np.array([u, v])


def camera_ray(u, v):
    return np.array([u / lam, v / lam, 1.0])


# 6. Fra kamera til robotbase
def ray_to_world(ray, T0C, z_plane):
    R = T0C[:3, :3]
    p = T0C[:3, 3]
    d = R @ ray
    if abs(d[2]) < 1e-9:
        raise ValueError('Synsstrålen er parallel med planet')
    s = (z_plane - p[2]) / d[2]
    if s <= 0:
        raise ValueError('Planet ligger bag kameraet')
    return p + s * d


# 7. Offline-test af IK, FK og baneplanlægning
def check_kinematics():
    q0 = IK(*p_test0)
    q1 = IK(*p_test1)
    p_check = FK(q1)[4, :3, 3]
    print('\nTest af kinematik (ikke fra kameraet):')
    print('  Mål [mm]:', p_test1)
    print('  Ledvinkler [grader]:', np.round(np.degrees(q1), 2))
    print('  Kontrol med FK [mm]:', np.round(p_check, 2))

    coeff = make_traj(q0, q1, duration=2.0)
    times, qs = sample_traj(coeff, duration=2.0)
    print(f'  Beregnet bane: {len(times)} punkter på {times[-1]:.1f} s')
    print('  Første og sidste vinkel [grader]:')
    print(' ', np.round(np.degrees(qs[0]), 2))
    print(' ', np.round(np.degrees(qs[-1]), 2))
    move(q1)  # robot.py udskriver kun; sender ingen motorkommandoer


# 8. Hovedprogram
def main():
    img = get_image()
    mask = segment_red(img)
    centers, marked, clean_mask = find_smarties(img, mask)
    print(f'Antal røde Smarties: {len(centers)}')

    rays = []
    smarties = []

    for i, (r, c) in enumerate(centers, start=1):
        u, v = pixel_to_image(r, c)
        ray = camera_ray(u, v)
        rays.append(ray)
        print(f'\nSmartie {i}: r={r:.1f}, c={c:.1f}')
        print(f'  Billedplan: u={u:.3f}, v={v:.3f} mm')
        print(f'  Synsstråle: {np.round(ray, 3)}')

        if T0C is not None and z_plane is not None:
            # Kun beregning: ingen bevægelse, heller ikke i robot-tilstand.
            pos = ray_to_world(ray, T0C, z_plane)
            smarties.append(pos)

            print('  Position i robotbase [mm]:', np.round(pos, 2))
            try:
                q = IK(*pos)
                print('  Ledvinkler [grader]:', np.round(np.degrees(q), 2))
            except ValueError as e:
                print('  IK kunne ikke beregnes:', e)

    if T0C is None or z_plane is None:
        print('\nRobotpositioner springes over: kamera og målplan er ikke kalibreret.')

    if test_ik:
        check_kinematics()

    # Simuler bevægelsen til de fundne Smarties
    if vis_sim and len(smarties) > 0:
        simuler(np.array(smarties), z_plane)

    # Vis billederne
    if show_img:
        cv2.imshow('Originalt billede', img)
        cv2.imshow('Rød farvemaske', clean_mask)
        cv2.imshow('Fundne Smarties', marked)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
