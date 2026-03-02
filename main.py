import cv2
import numpy as np


min_width = 80  # Minimum rectangle width
min_height = 80  # Minimum rectangle height

offset = 6  # Allowed pixel error

line_position = 550  # Counting line position

delay = 60  # Video FPS

detections = []
vehicles = 0


def get_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture('highway2.mp4')
background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

cv2.namedWindow("Video Original", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video Original", 1000, 600)


while True:
    ret, frame = cap.read()
    # if not ret:
    #   print("Frame not captured!")
    #   break

    tempo = float(1 / delay)
    sleep(tempo)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    foreground_mask = background_subtractor.apply(blur)

    dilated = cv2.dilate(foreground_mask, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    processed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    contours, h = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame, (25, line_position), (1200, line_position), (255, 127, 0), 3)

    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)

        valid_contour = (w >= min_width) and (h >= min_height)
        if not valid_contour:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        center = get_center(x, y, w, h)
        detections.append(center)

        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        for (x, y) in detections:
            if y < (line_position + offset) and y > (line_position - offset):
                vehicles += 1
                cv2.line(frame, (25, line_position), (1200, line_position), (0, 127, 255), 3)
                detections.remove((x, y))
                print("car is detected : " + str(vehicles))

    cv2.putText(frame,
                "VEHICLE COUNT : " + str(vehicles),
                (450, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                5)

    cv2.imshow("Video Original", frame)
    cv2.imshow("Detectar", processed)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()