from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import cv2
import os
from ultralytics import YOLO
import uuid
from fastapi.responses import HTMLResponse
from PIL import Image
import io

app = FastAPI(title="YOLO Object Detection API")

# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Mount static directory for serving result images
app.mount("/results", StaticFiles(directory="results"), name="results")

# Load YOLO model
try:
    model = YOLO("yolov8n.pt")  # You can change this to your trained model path
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/detect/")
async def detect_objects(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model failed to load")
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only PNG and JPEG images are supported")
    
    # Read and save the file
    contents = await file.read()
    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Generate unique filenames
    input_filename = f"uploads/{uuid.uuid4()}.jpg"
    output_filename = f"results/{uuid.uuid4()}.jpg"
    
    # Save input image
    cv2.imwrite(input_filename, image)
    
    # Run prediction
    results = model(image)
    
    # Process results (draw bounding boxes)
    result_image = results[0].plot()
    cv2.imwrite(output_filename, result_image)
    
    # Extract detected objects information
    boxes = results[0].boxes
    detection_results = []
    
    for box in boxes:
        # Get box coordinates, confidence and class
        coords = box.xyxy[0].tolist()  # Convert from tensor to list
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        detection_results.append({
            "class": class_name,
            "confidence": confidence,
            "bbox": coords
        })
    
    return {
        "result_image_url": f"/results/{os.path.basename(output_filename)}",
        "detections": detection_results
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)