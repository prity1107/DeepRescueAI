import cv2
import os
import uuid
import numpy as np
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("yolov8n.pt")  

os.makedirs("static", exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded', 'status': 'failed'}), 400
        
        file = request.files['image']
        
        # ✅ Validate file extension
        if not file.filename.lower().endswith('.jpg'):
            return jsonify({'error': '❌ Only JPG files are supported!', 'status': 'failed'}), 400

        # ✅ Save input image
        image_id = uuid.uuid4().hex  
        image_path = f"static/input_{image_id}.jpg"
        output_path = f"static/output_{image_id}.jpg"
        file.save(image_path)

        # ✅ Run YOLO object detection
        results = model(image_path)

        detected_objects = []
        img = cv2.imread(image_path)

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0]) * 100  # Convert to percentage
                detected_objects.append(f"{result.names[cls]} ({conf:.2f}%)")

                # ✅ Draw bounding boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{result.names[cls]}: {conf:.2f}%"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ✅ Save output image with detections
        cv2.imwrite(output_path, img)

        return jsonify({
            'status': 'success',
            'image': os.path.basename(output_path),
            'objects': ', '.join(detected_objects) if detected_objects else "No objects detected"
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")  # ✅ Log errors to console
        return jsonify({'error': str(e), 'status': 'failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
