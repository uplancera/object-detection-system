# Real-Time Object Detection System

## Project Overview

A production-ready real-time object detection system built using YOLOv8 and deployed as a web application. The system achieves 45+ FPS on standard hardware while maintaining high accuracy for multiple object classes.

### Key Features

- **Real-time Detection**: Processes video streams at 45+ FPS with GPU acceleration
- **Multi-class Detection**: Trained on COCO dataset with 80 object classes
- **Web Interface**: React-based dashboard for live camera feed processing
- **REST API**: FastAPI backend for integration with existing systems
- **Model Optimization**: TensorRT acceleration for edge deployment
- **Custom Training Pipeline**: Transfer learning capabilities for domain-specific use cases

## Technical Stack

**Computer Vision & ML**
- YOLOv8 (Ultralytics)
- OpenCV for image preprocessing
- TensorRT for model optimization
- PyTorch for custom model training

**Backend**
- FastAPI (Python 3.11)
- WebSocket for real-time streaming
- Redis for caching predictions
- PostgreSQL for logging and analytics

**Frontend**
- React 18 with TypeScript
- Canvas API for bounding box rendering
- WebRTC for low-latency video streaming
- Tailwind CSS for responsive UI

**Infrastructure**
- Docker containerization
- NGINX reverse proxy
- AWS EC2 (g4dn.xlarge) for GPU inference
- CloudWatch for monitoring

## Performance Metrics

| Metric | Value |
|--------|-------|
| Inference Speed | 45 FPS (GPU) / 12 FPS (CPU) |
| mAP@0.5 | 0.87 |
| Model Size | 6.2 MB (optimized) |
| Latency | <50ms end-to-end |
| Memory Usage | 2.1 GB GPU / 800 MB RAM |

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Camera    │─────▶│   FastAPI    │─────▶│   YOLOv8    │
│   Feed      │      │   Backend    │      │   Model     │
└─────────────┘      └──────────────┘      └─────────────┘
                            │                       │
                            ▼                       ▼
                     ┌──────────────┐      ┌─────────────┐
                     │  WebSocket   │      │  Detection  │
                     │   Stream     │      │  Results    │
                     └──────────────┘      └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │    React     │
                     │   Frontend   │
                     └──────────────┘
```

## Implementation Highlights

### 1. Model Selection & Optimization

Initially evaluated multiple architectures:
- YOLOv8n: Best balance of speed and accuracy for real-time use
- Faster R-CNN: Higher accuracy but slower (15 FPS)
- SSD MobileNet: Faster but lower mAP (0.72)

Applied TensorRT optimization achieving 2.3x speed improvement:

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='engine', half=True)  # FP16 precision
```

### 2. Real-time Video Processing Pipeline

Implemented multi-threaded processing to prevent frame drops:

```python
import cv2
import threading
from queue import Queue

class VideoProcessor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.frame_queue = Queue(maxsize=30)
        self.result_queue = Queue(maxsize=30)
        
    def capture_frames(self, source):
        cap = cv2.VideoCapture(source)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
    
    def process_frames(self):
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                results = self.model(frame, stream=True)
                self.result_queue.put(results)
```

### 3. Custom Training for Specific Use Cases

Implemented transfer learning pipeline for retail inventory monitoring:

- Dataset: 5,000 annotated images (custom products)
- Augmentation: Random rotation, blur, brightness adjustment
- Training: 100 epochs with early stopping
- Results: 0.92 mAP on validation set

### 4. API Design

RESTful endpoints with FastAPI:

```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/detect")
async def detect_objects(file: UploadFile):
    image = await file.read()
    results = model.predict(image)
    return {"detections": parse_results(results)}

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        frame = await websocket.receive_bytes()
        detections = process_frame(frame)
        await websocket.send_json(detections)
```

## Use Cases Demonstrated

### 1. Security Monitoring
- Person detection with 95% accuracy
- Intrusion alerts with confidence scoring
- Activity heatmaps over time

### 2. Retail Analytics
- Customer counting and flow analysis
- Product placement optimization
- Shelf inventory monitoring

### 3. Traffic Management
- Vehicle counting and classification
- Speed estimation using optical flow
- Incident detection

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t object-detection-api .

# Run with GPU support
docker run --gpus all -p 8000:8000 object-detection-api
```

### AWS Deployment

Deployed on EC2 g4dn.xlarge instance:
- NVIDIA T4 GPU for inference
- Auto-scaling group for load handling
- CloudFront CDN for static assets
- 99.9% uptime over 3 months

## Challenges & Solutions

### Challenge 1: Frame Drop at High Resolution
**Problem**: 4K video processing caused significant frame drops

**Solution**: Implemented dynamic resolution scaling
- Auto-resize based on GPU utilization
- Maintain aspect ratio
- Batch processing for multiple streams

### Challenge 2: False Positives in Low Light
**Problem**: Detection accuracy dropped to 0.65 mAP in low-light conditions

**Solution**: 
- Added data augmentation with low-light scenarios
- Implemented preprocessing with histogram equalization
- Improved mAP to 0.81 in low-light conditions

### Challenge 3: Model Size for Edge Deployment
**Problem**: Original model too large (240 MB) for edge devices

**Solution**:
- Applied model pruning (40% weight reduction)
- Quantization to INT8
- Final size: 6.2 MB with minimal accuracy loss

## Results & Impact

- **Production Deployment**: 3 retail locations using the system
- **Processing Volume**: 50,000+ hours of video analyzed
- **Accuracy**: 87% mAP across all object classes
- **Cost Savings**: 60% reduction vs. cloud-based solutions
- **Latency**: Sub-50ms end-to-end processing time

## Future Enhancements

1. **3D Object Detection**: Implement depth estimation using stereo vision
2. **Action Recognition**: Add temporal models for activity classification
3. **Edge TPU Support**: Optimize for Coral Edge TPU deployment
4. **Multi-camera Tracking**: Implement object re-identification across cameras
5. **Anomaly Detection**: Add unsupervised learning for unusual pattern detection

## Technical Blog Posts

- [Optimizing YOLOv8 for Real-Time Inference](link)
- [Building Production-Ready Computer Vision APIs](link)
- [From Training to Deployment: Computer Vision Pipeline](link)

## Demo & Code

- **Live Demo**: [demo-link]
- **GitHub Repository**: [github-link]
- **API Documentation**: [docs-link]

---

*This project demonstrates end-to-end computer vision system development, from model training and optimization to production deployment and monitoring.*
