from ultralytics import YOLO
import cv2

# 1. Load your model
# Since best.pt is in the same folder, we just use its name
model = YOLO('best.pt')

# 2. Open the Live Webcam
# source="0" uses your laptop's camera
# show=True pops up a window so you can see the results
# conf=0.5 ignores anything the AI is less than 50% sure about
model.predict(source="0", show=True, conf=0.5)