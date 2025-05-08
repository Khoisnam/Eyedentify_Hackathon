import cv2 # type: ignore
import numpy as np # type: ignore

# Load the model
protoPath = "path_to_detector/deploy.prototxt"
modelPath = "path_to_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load the image
image = cv2.imread("test_image.jpg")
if image is None:
    print("[ERROR] Could not load image. Check the file path.")
    exit()

(h, w) = image.shape[:2]
print(f"[DEBUG] Image dimensions: {h}x{w}")

# Create a blob and perform forward pass
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

# Debugging: Print the number of detections
print(f"[DEBUG] Detections found: {detections.shape[2]}")

# Draw bounding boxes
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    print(f"[DEBUG] Detection {i}: Confidence = {confidence}")
    if confidence > 0.3:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)