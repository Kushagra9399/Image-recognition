from ultralytics import YOLO
import cv2

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Use your desired model, e.g., yolov8s.pt, yolov8m.pt

# Load an image
img_path = 'workplace.jpg'  # Replace with your image path
img = cv2.imread(img_path)

# Perform object detection
results = model(img)

# Check if results are returned as a list (in case of batch processing)
# We only need the first result, so we can access results[0]
result = results[0]

# Get the result as a pandas dataframe
df = result.pandas().xywh  # Pandas dataframe with boxes in xywh format (center_x, center_y, width, height)

# Print the detected objects and their confidence scores
print("Detected objects:")
for index, row in df.iterrows():
    class_name = row['name']  # Get the class name
    confidence = row['confidence']  # Confidence score
    box = row[['x_center', 'y_center', 'width', 'height']]  # Bounding box (center_x, center_y, width, height)
    print(f"Class: {class_name}, Confidence: {confidence:.2f}, Box: {box.to_list()}")

# Display the result with bounding boxes using OpenCV
cv2.imshow('Detected Image', result.plot())  # Render the result and show the image

# Wait for keypress to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
