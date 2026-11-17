
import os
import cv2
import numpy as np

os.system('cls')

# Test af kamera og genkendelse af røde Smarties

# 1. Indstillinger

cam_id = 0          # Kameraets nummer (prøv 1 eller 2, hvis 0 ikke virker)

# HSV-grænser for rød
red1_low  = np.array([0, 130, 90])
red1_high = np.array([6, 255, 255])
red2_low  = np.array([174, 130, 90])
red2_high = np.array([179, 255, 255])

# Parametre til objektgenkendelse
min_area = 1000
min_circ = 0.6
k_open = 3
k_close = 11


# 2. Find røde Smarties

def find_smarties(img):

    # Konverter kamerabilledet til HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, red1_low, red1_high)
    mask2 = cv2.inRange(hsv, red2_low, red2_high)
    mask = cv2.bitwise_or(mask1, mask2)

    # Fjern støj
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (k_open, k_open)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Luk små huller
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (k_close, k_close)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    marked = img.copy()
    clean_mask = np.zeros_like(mask)
    centers = []

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

        # Marker Smartiens centrum
        pos = (round(c), round(r))
        cv2.circle(marked, pos, 12, (0, 0, 0), -1)
        cv2.circle(marked, pos, 9, (0, 255, 0), -1)

        cv2.drawContours(
            clean_mask, [contour], -1, 255, cv2.FILLED
        )

    return centers, marked, clean_mask


# 3. Start kameraet

cam = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)

if not cam.isOpened():
    raise RuntimeError(
        "Kameraet kunne ikke åbnes. Prøv cam_id = 1 eller 2."
    )

print("Kameraet er tilsluttet.")
print("Tryk Q for at afslutte.")


# 4. Læs og behandl billeder løbende

try:
    while True:

        success, img = cam.read()

        if not success:
            print("Kunne ikke hente billede fra kameraet.")
            break

        centers, marked, mask = find_smarties(img)

        # Vis antal fundne Smarties
        cv2.putText(
            marked,
            f"Rode Smarties: {len(centers)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow("Kamera - Rode Smarties", marked)
        cv2.imshow("Rod farvemaske", mask)

        # Tryk Q for at stoppe
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cam.release()
    cv2.destroyAllWindows()
