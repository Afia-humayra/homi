import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width, _ = frame.shape

    cx = int(width / 2)
    cy = int(height / 2)

    pixel_center = hsv_frame[cy, cx]
    hue_value = pixel_center[0]
    s = pixel_center[1]
    v = pixel_center[2]

    # Determine color
    #color = "Undefined"
    if v <= 50:
        color = "Black"
    elif s <= 50 and v >= 200:
        color = "White"
    elif s <= 50:
        color = "Gray"
    else:
        if hue_value < 5 or hue_value >= 178:
            color = "Red"
        elif hue_value < 22:
            color = "Orange"
        elif hue_value < 33:
            color = "Yellow"
        elif hue_value < 78:
            color = "Green"
        elif hue_value < 131:
            color = "Blue"
        elif hue_value < 178:
            color = "Violet"
        else:
            color = "Red"



    pixel_center_bgr = hsv_frame[cy, cx]
    cv2.putText(frame, color, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), 3)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()