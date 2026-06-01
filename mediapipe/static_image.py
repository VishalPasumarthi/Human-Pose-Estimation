import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Drawing utility
mp_draw = mp.solutions.drawing_utils

# Read image
image = cv2.imread("images/person.png")


# Convert BGR to RGB
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process image
results = pose.process(rgb)

# Draw landmarks
if results.pose_landmarks:
    mp_draw.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

# Show output
cv2.imshow("Pose Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
