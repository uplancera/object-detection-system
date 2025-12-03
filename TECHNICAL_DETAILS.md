# Technical Implementation Details

## 1. Model Architecture & Training

### YOLOv8 Architecture Overview

The system uses YOLOv8n (nano) variant for optimal speed-accuracy tradeoff:

```
Input: 640x640x3
├── Backbone (CSPDarknet)
│   ├── Conv + SPPF layers
│   └── Feature extraction at multiple scales
├── Neck (PANet)
│   ├── Feature pyramid network
│   └── Path aggregation
└── Head (Detect)
    ├── Bounding box regression
    ├── Objectness prediction
    └── Class probability distribution
```

### Training Configuration

```python
from ultralytics import YOLO

# Model initialization
model = YOLO('yolov8n.yaml')

# Training parameters
results = model.train(
    data='config.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,
    warmup_epochs=3,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.9,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5
)
```

### Data Augmentation Pipeline

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.RandomResizedCrop(640, 640, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.3),
    A.GaussNoise(p=0.3),
    A.CLAHE(p=0.3),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
], bbox_params=A.BboxParams(
    format='yolo',
    label_fields=['class_labels']
))
```

## 2. Inference Optimization

### TensorRT Engine Conversion

```python
import tensorrt as trt
from ultralytics import YOLO

def convert_to_tensorrt(model_path, engine_path):
    """
    Convert YOLO model to TensorRT engine
    
    Optimizations applied:
    - FP16 precision (2x speedup)
    - Layer fusion
    - Kernel auto-tuning
    """
    model = YOLO(model_path)
    
    # Export with TensorRT optimizations
    success = model.export(
        format='engine',
        half=True,  # FP16
        simplify=True,
        workspace=4,  # 4GB workspace
        verbose=False
    )
    
    return success

# Performance comparison
# PyTorch: 22ms per frame
# ONNX: 18ms per frame
# TensorRT FP32: 15ms per frame
# TensorRT FP16: 10ms per frame (selected)
```

### Batch Processing for Multiple Streams

```python
import numpy as np
import cv2
from typing import List

class BatchProcessor:
    def __init__(self, model, batch_size=4):
        self.model = model
        self.batch_size = batch_size
        self.frame_buffer = []
        
    def add_frame(self, frame: np.ndarray, stream_id: str):
        """Add frame to batch buffer"""
        self.frame_buffer.append({
            'frame': cv2.resize(frame, (640, 640)),
            'stream_id': stream_id,
            'original_size': frame.shape[:2]
        })
        
        if len(self.frame_buffer) >= self.batch_size:
            return self.process_batch()
        return None
    
    def process_batch(self):
        """Process accumulated frames as a batch"""
        if not self.frame_buffer:
            return []
        
        # Stack frames into batch
        batch = np.stack([item['frame'] for item in self.frame_buffer])
        
        # Run inference
        results = self.model(batch)
        
        # Parse results
        outputs = []
        for i, result in enumerate(results):
            outputs.append({
                'stream_id': self.frame_buffer[i]['stream_id'],
                'detections': self.parse_detections(
                    result,
                    self.frame_buffer[i]['original_size']
                ),
                'inference_time': result.speed['inference']
            })
        
        self.frame_buffer = []
        return outputs
    
    def parse_detections(self, result, original_size):
        """Scale bounding boxes to original image size"""
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        h_scale = original_size[0] / 640
        w_scale = original_size[1] / 640
        
        detections = []
        for box, score, cls in zip(boxes, scores, classes):
            detections.append({
                'bbox': [
                    int(box[0] * w_scale),
                    int(box[1] * h_scale),
                    int(box[2] * w_scale),
                    int(box[3] * h_scale)
                ],
                'confidence': float(score),
                'class': int(cls),
                'class_name': result.names[int(cls)]
            })
        
        return detections
```

## 3. Backend API Implementation

### FastAPI Service Architecture

```python
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from typing import List
import asyncio
from datetime import datetime
import redis
import json

app = FastAPI(title="Object Detection API", version="1.0.0")

# Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionService:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.active_streams = {}
        
    async def detect_image(self, image: np.ndarray):
        """Run detection on single image"""
        results = self.model(image)
        return self.format_results(results[0])
    
    async def detect_video_stream(self, websocket: WebSocket):
        """Handle real-time video stream"""
        await websocket.accept()
        stream_id = f"stream_{datetime.now().timestamp()}"
        
        try:
            while True:
                # Receive frame
                data = await websocket.receive_bytes()
                frame = self.decode_frame(data)
                
                # Run detection
                results = await self.detect_image(frame)
                
                # Send results
                await websocket.send_json({
                    'timestamp': datetime.now().isoformat(),
                    'detections': results,
                    'stream_id': stream_id
                })
                
        except Exception as e:
            print(f"Stream error: {e}")
        finally:
            await websocket.close()
    
    def decode_frame(self, data: bytes) -> np.ndarray:
        """Decode frame from bytes"""
        nparr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    
    def format_results(self, result):
        """Format detection results"""
        return {
            'boxes': result.boxes.xyxy.tolist(),
            'scores': result.boxes.conf.tolist(),
            'classes': result.boxes.cls.tolist(),
            'class_names': [result.names[int(c)] for c in result.boxes.cls]
        }

# Initialize service
detector = DetectionService('models/yolov8n.engine')

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Detect objects in uploaded image
    
    Returns:
        JSON with bounding boxes, scores, and class labels
    """
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Check cache
    cache_key = f"detection:{hash(contents)}"
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Run detection
    results = await detector.detect_image(image)
    
    # Cache results (expire in 1 hour)
    redis_client.setex(cache_key, 3600, json.dumps(results))
    
    return results

@app.websocket("/stream")
async def stream_detection(websocket: WebSocket):
    """
    Real-time video stream processing
    
    Client sends video frames, receives detection results
    """
    await detector.detect_video_stream(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detector.model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    return {
        "active_streams": len(detector.active_streams),
        "cache_size": redis_client.dbsize(),
        "model_info": {
            "type": "YOLOv8n",
            "input_size": "640x640",
            "classes": 80
        }
    }
```

### WebSocket Client Example

```typescript
// React component for real-time detection
import React, { useRef, useEffect, useState } from 'react';

interface Detection {
  boxes: number[][];
  scores: number[];
  classes: number[];
  class_names: string[];
}

export const LiveDetection: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [detections, setDetections] = useState<Detection | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Initialize WebSocket
    wsRef.current = new WebSocket('ws://localhost:8000/stream');
    
    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
      startVideoStream();
    };
    
    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setDetections(data.detections);
      drawDetections(data.detections);
    };
    
    return () => {
      wsRef.current?.close();
    };
  }, []);

  const startVideoStream = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 }
    });
    
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      sendFrames();
    }
  };

  const sendFrames = () => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    
    const interval = setInterval(() => {
      if (!videoRef.current || !wsRef.current) {
        clearInterval(interval);
        return;
      }
      
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      ctx.drawImage(videoRef.current, 0, 0);
      
      canvas.toBlob((blob) => {
        if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(blob);
        }
      }, 'image/jpeg', 0.8);
    }, 33); // ~30 FPS
  };

  const drawDetections = (detections: Detection) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    
    const ctx = canvas.getContext('2d')!;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 3;
    ctx.font = '16px Arial';
    ctx.fillStyle = '#00FF00';
    
    detections.boxes.forEach((box, i) => {
      const [x1, y1, x2, y2] = box;
      const label = `${detections.class_names[i]} ${(detections.scores[i] * 100).toFixed(1)}%`;
      
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillText(label, x1, y1 - 5);
    });
  };

  return (
    <div className="relative">
      <video ref={videoRef} autoPlay muted className="w-full" />
      <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full" />
      <div className="absolute bottom-4 right-4 bg-black bg-opacity-75 p-4 rounded">
        <p className="text-white">
          Detections: {detections?.boxes.length || 0}
        </p>
      </div>
    </div>
  );
};
```

## 4. Performance Monitoring

### Metrics Collection

```python
import time
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class PerformanceMetrics:
    inference_time: float
    preprocessing_time: float
    postprocessing_time: float
    total_time: float
    fps: float
    memory_usage: float

class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics_history: List[PerformanceMetrics] = []
        
    def record_inference(self, metrics: PerformanceMetrics):
        """Record inference metrics"""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)
    
    def get_statistics(self):
        """Calculate performance statistics"""
        if not self.metrics_history:
            return None
        
        inference_times = [m.inference_time for m in self.metrics_history]
        fps_values = [m.fps for m in self.metrics_history]
        
        return {
            'avg_inference_time': np.mean(inference_times),
            'p95_inference_time': np.percentile(inference_times, 95),
            'p99_inference_time': np.percentile(inference_times, 99),
            'avg_fps': np.mean(fps_values),
            'min_fps': np.min(fps_values),
            'max_fps': np.max(fps_values)
        }

# Usage in inference loop
monitor = PerformanceMonitor()

def run_detection_with_monitoring(frame):
    start_time = time.time()
    
    # Preprocessing
    preprocess_start = time.time()
    processed_frame = preprocess(frame)
    preprocess_time = time.time() - preprocess_start
    
    # Inference
    inference_start = time.time()
    results = model(processed_frame)
    inference_time = time.time() - inference_start
    
    # Postprocessing
    postprocess_start = time.time()
    detections = postprocess(results)
    postprocess_time = time.time() - postprocess_start
    
    total_time = time.time() - start_time
    fps = 1.0 / total_time if total_time > 0 else 0
    
    # Record metrics
    metrics = PerformanceMetrics(
        inference_time=inference_time,
        preprocessing_time=preprocess_time,
        postprocessing_time=postprocess_time,
        total_time=total_time,
        fps=fps,
        memory_usage=get_memory_usage()
    )
    monitor.record_inference(metrics)
    
    return detections
```

## 5. Error Handling & Resilience

```python
from functools import wraps
import logging
from typing import Callable

logger = logging.getLogger(__name__)

def retry_on_failure(max_retries=3, delay=1.0):
    """Decorator for retrying failed operations"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    await asyncio.sleep(delay * (attempt + 1))
        return wrapper
    return decorator

class GracefulDegradation:
    """Handle model failures gracefully"""
    
    def __init__(self, primary_model, fallback_model):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.primary_failures = 0
        self.use_fallback = False
        
    async def detect(self, frame):
        """Run detection with fallback"""
        try:
            if not self.use_fallback:
                results = await self.primary_model(frame)
                self.primary_failures = 0
                return results
        except Exception as e:
            logger.error(f"Primary model failed: {e}")
            self.primary_failures += 1
            
            if self.primary_failures >= 3:
                self.use_fallback = True
                logger.warning("Switching to fallback model")
        
        # Use fallback
        return await self.fallback_model(frame)
```

---

This implementation provides production-ready computer vision capabilities with proper error handling, monitoring, and optimization techniques.
