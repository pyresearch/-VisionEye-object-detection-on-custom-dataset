import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import pyresearch

# Load the YOLO model
model = YOLO("last.pt")
names = model.model.names

# Set up video capture and get dimensions
cap = cv2.VideoCapture("demo1.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define video output parameters for MP4 format
out = cv2.VideoWriter("visioneye-pinpoint.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

center_point = (-10, h)

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Run YOLO prediction
    results = model.predict(im0)
    boxes = results[0].boxes.xyxy.cpu()
    clss = results[0].boxes.cls.cpu().tolist()

    # Annotate the frame
    annotator = Annotator(im0, line_width=2)
    for box, cls in zip(boxes, clss):
        annotator.box_label(box, label=names[int(cls)], color=colors(int(cls)))
        annotator.visioneye(box, center_point)

    # Write the annotated frame to output
    out.write(im0)
    
    # Resize the frame to 1080 pixels in width for display in the imshow window
    display_width = 1080
    display_height = int((display_width / w) * h)  # Calculate the corresponding height to maintain aspect ratio
    im_display = cv2.resize(im0, (display_width, display_height))

    # Show the resized frame
    cv2.imshow("visioneye-pinpoint", im_display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
out.release()
cap.release()
cv2.destroyAllWindows()
