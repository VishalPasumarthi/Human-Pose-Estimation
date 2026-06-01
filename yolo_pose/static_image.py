from ultralytics import YOLO
import cv2

# Load YOLO Pose model
model = YOLO("yolov8n-pose.pt")

# Read image
image = cv2.imread("images/person.png")

# Run pose detection
results = model(image)

# Draw results
annotated_image = results[0].plot()

# Show output
cv2.imshow("YOLO Pose Detection", annotated_image)

cv2.waitKey(0)
cv2.destroyAllWindows()