import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load MoveNet
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    height, width, _ = frame.shape

    # Prepare image
    img = cv2.resize(frame, (192, 192))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_image = tf.expand_dims(img, axis=0)
    input_image = tf.cast(input_image, dtype=tf.int32)

    # Inference
    outputs = movenet(input_image)
    keypoints = outputs['output_0'].numpy()

    # Draw points
    for kp in keypoints[0][0]:
        y = int(kp[0] * height)
        x = int(kp[1] * width)
        confidence = kp[2]

        if confidence > 0.2:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("MoveNet Realtime Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()