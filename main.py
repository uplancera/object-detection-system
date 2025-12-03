"""
Real-Time Object Detection System
Main application file with API endpoints and detection logic
"""

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
from typing import List, Dict, Optional
import asyncio
import time
from datetime import datetime
import logging
from pathlib import Path
import redis
import json
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Histogram, generate_latest
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
DETECTION_COUNTER = Counter('detections_total', 'Total number of detections')
INFERENCE_TIME = Histogram('inference_duration_seconds', 'Inference time in seconds')
REQUEST_COUNTER = Counter('api_requests_total', 'Total API requests', ['endpoint'])

@dataclass
class Detection:
    """Data class for detection results"""
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ObjectDetector:
    """Core object detection class"""
    
    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        device: str = "cuda:0",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        """
        Initialize object detector
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to run inference on (cuda:0 or cpu)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        logger.info(f"Loading model from {model_path}")
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = self.model.names
        
        # Warm up model
        logger.info("Warming up model...")
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy_image, device=device, verbose=False)
        logger.info("Model ready")
    
    def detect(
        self,
        image: np.ndarray,
        return_image: bool = False
    ) -> Dict:
        """
        Run object detection on image
        
        Args:
            image: Input image as numpy array (HxWxC)
            return_image: Whether to return annotated image
            
        Returns:
            Dictionary containing detections and metadata
        """
        start_time = time.time()
        
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        # Parse results
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, cls in zip(boxes, scores, classes):
                detection = Detection(
                    bbox=box.tolist(),
                    confidence=float(score),
                    class_id=int(cls),
                    class_name=self.class_names[cls]
                )
                detections.append(asdict(detection))
                DETECTION_COUNTER.inc()
        
        inference_time = time.time() - start_time
        INFERENCE_TIME.observe(inference_time)
        
        response = {
            'detections': detections,
            'count': len(detections),
            'inference_time_ms': round(inference_time * 1000, 2),
            'image_shape': image.shape[:2]
        }
        
        if return_image:
            annotated = results.plot()
            response['annotated_image'] = annotated
        
        return response
    
    def detect_batch(
        self,
        images: List[np.ndarray]
    ) -> List[Dict]:
        """
        Run detection on batch of images
        
        Args:
            images: List of images as numpy arrays
            
        Returns:
            List of detection results
        """
        start_time = time.time()
        
        results = self.model(
            images,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        batch_results = []
        for result in results:
            detections = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, score, cls in zip(boxes, scores, classes):
                    detection = Detection(
                        bbox=box.tolist(),
                        confidence=float(score),
                        class_id=int(cls),
                        class_name=self.class_names[cls]
                    )
                    detections.append(asdict(detection))
            
            batch_results.append({
                'detections': detections,
                'count': len(detections)
            })
        
        inference_time = time.time() - start_time
        logger.info(f"Batch inference time: {inference_time:.3f}s for {len(images)} images")
        
        return batch_results


class VideoProcessor:
    """Process video streams for real-time detection"""
    
    def __init__(self, detector: ObjectDetector):
        self.detector = detector
        self.active_streams = {}
        
    async def process_stream(
        self,
        websocket: WebSocket,
        stream_id: str
    ):
        """
        Process real-time video stream from WebSocket
        
        Args:
            websocket: WebSocket connection
            stream_id: Unique identifier for stream
        """
        await websocket.accept()
        self.active_streams[stream_id] = True
        logger.info(f"Stream {stream_id} started")
        
        try:
            while self.active_streams.get(stream_id, False):
                # Receive frame
                data = await websocket.receive_bytes()
                
                # Decode frame
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.warning("Failed to decode frame")
                    continue
                
                # Run detection
                results = self.detector.detect(frame)
                
                # Send results back
                await websocket.send_json({
                    'stream_id': stream_id,
                    'timestamp': datetime.now().isoformat(),
                    'results': results
                })
                
        except Exception as e:
            logger.error(f"Stream {stream_id} error: {e}")
        finally:
            self.active_streams[stream_id] = False
            await websocket.close()
            logger.info(f"Stream {stream_id} ended")


# Initialize Redis for caching
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connected")
except Exception as e:
    logger.warning(f"Redis not available: {e}")
    redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for FastAPI app"""
    # Startup
    logger.info("Starting up application")
    global detector, video_processor
    
    detector = ObjectDetector(
        model_path="models/yolov8n.pt",
        device="cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    )
    video_processor = VideoProcessor(detector)
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")


# Initialize FastAPI app
app = FastAPI(
    title="Object Detection API",
    description="Real-time object detection using YOLOv8",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    REQUEST_COUNTER.labels(endpoint='root').inc()
    return {
        "message": "Object Detection API",
        "version": "1.0.0",
        "endpoints": {
            "detect": "/detect",
            "stream": "/stream",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    REQUEST_COUNTER.labels(endpoint='health').inc()
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": detector is not None,
        "device": detector.device if detector else "unknown",
        "redis_available": redis_client is not None
    }
    
    return health_status


@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    conf_threshold: Optional[float] = None,
    return_image: bool = False
):
    """
    Detect objects in uploaded image
    
    Args:
        file: Image file (JPEG, PNG)
        conf_threshold: Override confidence threshold
        return_image: Return annotated image
        
    Returns:
        JSON with detection results
    """
    REQUEST_COUNTER.labels(endpoint='detect').inc()
    
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Check cache
        cache_key = f"detection:{hash(contents)}"
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info("Returning cached result")
                return json.loads(cached)
        
        # Override confidence threshold if provided
        original_conf = detector.conf_threshold
        if conf_threshold is not None:
            detector.conf_threshold = conf_threshold
        
        # Run detection
        results = detector.detect(image, return_image=return_image)
        
        # Restore original threshold
        detector.conf_threshold = original_conf
        
        # Cache results
        if redis_client:
            redis_client.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(results, default=str)
            )
        
        return results
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/stream")
async def stream_detection(websocket: WebSocket):
    """
    Real-time video stream detection via WebSocket
    
    Client sends video frames, receives detection results
    """
    stream_id = f"stream_{datetime.now().timestamp()}"
    REQUEST_COUNTER.labels(endpoint='stream').inc()
    
    await video_processor.process_stream(websocket, stream_id)


@app.post("/detect/batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    """
    Detect objects in multiple images
    
    Args:
        files: List of image files
        
    Returns:
        List of detection results
    """
    REQUEST_COUNTER.labels(endpoint='detect_batch').inc()
    
    try:
        images = []
        for file in files:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:
                images.append(image)
        
        if not images:
            raise HTTPException(status_code=400, detail="No valid images provided")
        
        results = detector.detect_batch(images)
        
        return {
            "batch_size": len(images),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


@app.get("/stats")
async def get_statistics():
    """Get detection statistics"""
    REQUEST_COUNTER.labels(endpoint='stats').inc()
    
    stats = {
        "total_detections": DETECTION_COUNTER._value.get(),
        "active_streams": len([s for s in video_processor.active_streams.values() if s]),
        "model_info": {
            "name": "YOLOv8n",
            "device": detector.device,
            "classes": len(detector.class_names),
            "conf_threshold": detector.conf_threshold,
            "iou_threshold": detector.iou_threshold
        }
    }
    
    return stats


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
