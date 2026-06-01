import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load MoveNet model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

# Read image
image = cv2.imread("../images/person.png")

# Check image loaded
if image is None:
    print("Image not found!")
    exit()

# Resize image for MoveNet
img = cv2.resize(image, (192, 192))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_image = tf.expand_dims(img, axis=0)
input_image = tf.cast(input_image, dtype=tf.int32)

# Run inference
outputs = movenet(input_image)
keypoints = outputs['output_0'].numpy()

# Draw keypoints
height, width, _ = image.shape

for kp in keypoints[0][0]:
    y = int(kp[0] * height)
    x = int(kp[1] * width)
    confidence = kp[2]

    if confidence > 0.2:
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

# Save output
cv2.imwrite("movenet_output.png", image)

print("Output saved as movenet_output.png")