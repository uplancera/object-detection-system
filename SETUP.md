# Setup & Installation Guide

## Prerequisites

### System Requirements

**Minimum Requirements:**
- Python 3.9+
- 8GB RAM
- 4GB GPU VRAM (NVIDIA GPU recommended)
- 10GB disk space

**Recommended Requirements:**
- Python 3.11
- 16GB RAM
- 8GB GPU VRAM (RTX 3060 or better)
- 50GB disk space (for training)
- Ubuntu 22.04 or Windows 11

### Software Dependencies

- CUDA 11.8+ (for GPU acceleration)
- cuDNN 8.6+
- Docker 20.10+ (for containerized deployment)
- Node.js 18+ (for frontend)

## Installation

### Option 1: Local Development Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/uplancera/object-detection-system.git
cd object-detection-system
```

#### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n objdet python=3.11
conda activate objdet
```

#### 3. Install Python Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
ultralytics==8.1.0
opencv-python==4.9.0.80
numpy==1.24.3
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
websockets==12.0
redis==5.0.1
pydantic==2.5.3
Pillow==10.2.0
albumentations==1.3.1
tensorrt==8.6.1
onnx==1.15.0
onnxruntime-gpu==1.16.3
prometheus-client==0.19.0
python-dotenv==1.0.0
```

#### 4. Download Pre-trained Models

```bash
# Download YOLOv8 weights
python scripts/download_models.py

# Or manually
mkdir models
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
cd ..
```

#### 5. Set Up Configuration

```bash
# Copy example config
cp config.example.yaml config.yaml

# Edit configuration
nano config.yaml
```

**config.yaml:**
```yaml
model:
  type: "yolov8n"
  weights: "models/yolov8n.pt"
  confidence: 0.5
  iou_threshold: 0.45
  device: "cuda:0"  # or "cpu"
  
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  
redis:
  host: "localhost"
  port: 6379
  db: 0
  
logging:
  level: "INFO"
  file: "logs/app.log"
```

#### 6. Run the Application

```bash
# Start Redis (if not already running)
redis-server &

# Start backend API
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, start frontend (if applicable)
cd frontend
npm install
npm run dev
```

### Option 2: Docker Deployment

#### 1. Build Docker Image

```bash
# Build image
docker build -t object-detection:latest .

# Or use docker-compose
docker-compose build
```

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download models
RUN python3 scripts/download_models.py

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - api
    restart: unless-stopped

volumes:
  redis_data:
```

#### 2. Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 3: Cloud Deployment (AWS)

#### 1. Prepare AWS Environment

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
```

#### 2. Deploy with AWS CDK

```bash
# Install CDK
npm install -g aws-cdk

# Deploy stack
cd infrastructure
cdk deploy ObjectDetectionStack
```

#### 3. Or Use Terraform

```bash
# Initialize Terraform
cd terraform
terraform init

# Plan deployment
terraform plan

# Apply configuration
terraform apply
```

**terraform/main.tf (excerpt):**
```hcl
resource "aws_instance" "detection_server" {
  ami           = "ami-0c55b159cbfafe1f0"  # Deep Learning AMI
  instance_type = "g4dn.xlarge"
  
  tags = {
    Name = "ObjectDetectionServer"
  }
  
  user_data = <<-EOF
    #!/bin/bash
    git clone https://github.com/uplancera/object-detection-system.git
    cd object-detection-system
    docker-compose up -d
  EOF
}
```

## Configuration

### Model Configuration

#### Using Different YOLO Variants

```python
# In config.yaml or programmatically
models = {
    'nano': 'yolov8n.pt',      # Fastest, 6MB
    'small': 'yolov8s.pt',     # Balanced, 22MB
    'medium': 'yolov8m.pt',    # Accurate, 52MB
    'large': 'yolov8l.pt',     # Very accurate, 87MB
    'xlarge': 'yolov8x.pt'     # Most accurate, 131MB
}
```

#### Custom Training

```bash
# Train on custom dataset
python train.py \
  --data custom_data.yaml \
  --model yolov8n.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

**custom_data.yaml:**
```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 10  # number of classes
names: ['class1', 'class2', ..., 'class10']
```

### Performance Tuning

#### GPU Optimization

```python
# In main.py or config
import torch

# Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set deterministic mode (if needed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
```

#### Batch Size Tuning

```python
# Find optimal batch size
def find_optimal_batch_size(model, img_size=640):
    batch_size = 1
    while True:
        try:
            dummy_input = torch.randn(batch_size, 3, img_size, img_size).cuda()
            _ = model(dummy_input)
            batch_size *= 2
        except RuntimeError:
            optimal_batch = batch_size // 2
            print(f"Optimal batch size: {optimal_batch}")
            return optimal_batch
```

## Testing

### Run Unit Tests

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Run tests
pytest tests/ -v --cov=.

# Run specific test file
pytest tests/test_detection.py -v
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/load_test.py --host http://localhost:8000
```

**tests/load_test.py:**
```python
from locust import HttpUser, task, between
import random

class DetectionUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def detect_image(self):
        with open('test_images/sample.jpg', 'rb') as f:
            self.client.post('/detect', files={'file': f})
    
    @task(2)
    def health_check(self):
        self.client.get('/health')
```

## Monitoring & Logging

### Prometheus Metrics

```bash
# Metrics endpoint available at
http://localhost:8000/metrics
```

### Grafana Dashboard

```bash
# Start Grafana
docker run -d -p 3001:3000 grafana/grafana

# Import dashboard
# Use provided dashboard JSON in monitoring/grafana_dashboard.json
```

### Log Aggregation

```bash
# Using ELK Stack
docker-compose -f docker-compose.elk.yml up -d

# View logs at
http://localhost:5601
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```python
# Solution: Reduce batch size or image size
# In config.yaml:
model:
  batch_size: 1
  imgsz: 320  # Reduced from 640
```

#### 2. Slow Inference on CPU

```bash
# Install OpenVINO for CPU optimization
pip install openvino-dev

# Convert model
mo --input_model yolov8n.onnx --output_dir openvino_model
```

#### 3. WebSocket Connection Issues

```python
# Increase timeout in frontend
const ws = new WebSocket('ws://localhost:8000/stream', {
  timeout: 30000,
  reconnectInterval: 3000
});
```

#### 4. Redis Connection Errors

```bash
# Check Redis status
redis-cli ping

# Restart Redis
sudo systemctl restart redis
```

### Debug Mode

```bash
# Run with debug logging
export LOG_LEVEL=DEBUG
python -m uvicorn main:app --reload --log-level debug
```

## Performance Benchmarks

### Benchmark Your Setup

```bash
# Run benchmark script
python scripts/benchmark.py \
  --model models/yolov8n.pt \
  --device cuda:0 \
  --runs 100
```

Expected results on RTX 3060:
```
Average FPS: 45.2
P50 latency: 22ms
P95 latency: 28ms
P99 latency: 35ms
Memory usage: 1.9GB
```

## Next Steps

1. **Customize for Your Use Case**
   - Collect domain-specific data
   - Fine-tune the model
   - Adjust confidence thresholds

2. **Integrate with Existing Systems**
   - Use REST API endpoints
   - Implement webhooks for events
   - Set up database logging

3. **Scale Your Deployment**
   - Set up load balancing
   - Implement auto-scaling
   - Add monitoring and alerts

## Support & Resources

- **Documentation**: [docs link]
- **GitHub Issues**: [issues link]
- **Discord Community**: [discord link]
- **API Reference**: [api-docs link]

---

For questions or issues, please open a GitHub issue or contact support@example.com
