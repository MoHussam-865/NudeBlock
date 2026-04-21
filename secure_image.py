import cv2
import numpy as np
import onnxruntime as ort

# --- 1. Configuration ---
MODEL_PATH = "nudenet_v8.onnx" # Replace with your ONNX file name
IMAGE_PATH = str(input("Enter the input video file name: "))  # Replace with your test image
CONF_THRESHOLD = 0.25          # Minimum confidence score
IOU_THRESHOLD = 0.45           # Threshold for overlapping boxes (NMS)

# The 18 classes supported by the deepghs NudeNet model
# The 18 classes supported by the model, in the correct index order
CLASSES = [
    "FEMALE_GENITALIA_COVERED", 
    "FACE_FEMALE", 
    "BUTTOCKS_EXPOSED", 
    "FEMALE_BREAST_EXPOSED", 
    "FEMALE_GENITALIA_EXPOSED", 
    "MALE_BREAST_EXPOSED", 
    "ANUS_EXPOSED", 
    "FEET_EXPOSED", 
    "BELLY_COVERED", 
    "FEET_COVERED", 
    "ARMPITS_COVERED", 
    "ARMPITS_EXPOSED", 
    "FACE_MALE", 
    "BELLY_EXPOSED", 
    "MALE_GENITALIA_EXPOSED", 
    "ANUS_COVERED", 
    "FEMALE_BREAST_COVERED", 
    "BUTTOCKS_COVERED"
]

# --- 2. Load Model ---
# Use CPU by default. Change to ['CUDAExecutionProvider'] if you have an Nvidia GPU setup
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# --- 3. Pre-process Image ---
# Load original image
original_image = cv2.imread(IMAGE_PATH)
original_h, original_w = original_image.shape[:2]

# YOLOv8 expects 640x640 input. We use cv2.dnn.blobFromImage for easy conversion
# This resizes, converts BGR to RGB (swapRB=True), and normalizes pixels to 0.0-1.0 (1/255.0)
input_tensor = cv2.dnn.blobFromImage(
    original_image, 
    1 / 255.0, 
    (640, 640), 
    swapRB=True, 
    crop=False
)

# --- 4. Run Inference ---
# The output shape of YOLOv8 ONNX is typically [1, 22, 8400] 
# (1 batch, 4 box coords + 18 classes = 22, 8400 anchor boxes)
outputs = session.run(None, {input_name: input_tensor})
predictions = outputs[0]

# --- 5. Post-process (Filter and NMS) ---
# Transpose to make looping easier: [1, 22, 8400] -> [8400, 22]
predictions = np.squeeze(predictions).T 

boxes = []
scores = []
class_ids = []

# Scaling factors to map 640x640 coordinates back to the original image size
x_factor = original_w / 640.0
y_factor = original_h / 640.0

for row in predictions:
    # row[0:4] are box coordinates (center_x, center_y, width, height)
    # row[4:] are class probabilities
    classes_scores = row[4:]
    max_score = np.max(classes_scores)
    
    if max_score >= CONF_THRESHOLD:
        class_id = np.argmax(classes_scores)
        
        # YOLOv8 returns bounding box center coordinates, width, and height
        cx, cy, w, h = row[0], row[1], row[2], row[3]
        
        # Convert to Top-Left x, y, width, height for OpenCV NMS
        left = int((cx - (w / 2)) * x_factor)
        top = int((cy - (h / 2)) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)
        
        boxes.append([left, top, width, height])
        scores.append(float(max_score))
        class_ids.append(class_id)

# Apply Non-Maximum Suppression to remove redundant overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)

# --- 6. Draw Boxes ---
if len(indices) > 0:
    for i in indices.flatten():
        box = boxes[i]
        left, top, width, height = box[0], box[1], box[2], box[3]
        score = scores[i]
        class_id = class_ids[i]
        label = CLASSES[class_id]
        
        # Draw the rectangle
        cv2.rectangle(original_image, (left, top), (left + width, top + height), (0, 0, 255), 2)
        
        # Draw label and score
        text = f"{label}: {score:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Create background for text
        cv2.rectangle(original_image, (left, top - 20), (left + text_size[0], top), (0, 0, 255), -1)
        cv2.putText(original_image, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# --- 7. Display ---
cv2.imshow("Detection Results", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()