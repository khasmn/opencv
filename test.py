import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    height, width, _ = frame.shape
    x_position = width // 2
    y_position = height // 2

    cv2.line(annotated_frame, (0, y_position), (width, y_position), (255, 0, 0), 2)
    cv2.line(annotated_frame, (x_position, 0), (x_position, height), (255, 0, 0), 2)

    phone_in_middle = False

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_center_x = (x1 + x2) // 2
        box_center_y = (y1 + y2) // 2

        margin = 50
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name.lower() == "cell phone":
            if abs(box_center_x - x_position) < margin and abs(box_center_y - y_position) < margin:
                phone_in_middle = True

    if phone_in_middle:
        cv2.putText(annotated_frame, "Cell Phone in Middle", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO - Cell Phone Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
