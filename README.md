# Suture Analysis API

A computer vision-based API for analyzing surgical sutures using YOLOv8 and OpenCV.

## Features

- FastAPI-based REST API
- YOLOv8 for object detection
- Multiple preprocessing options (grayscale, edge detection, contour detection)
- Docker-based deployment
- Health check endpoint
- Structured logging
- Environment-based configuration

## Project Structure

```
backend/
├── app/
│   ├── api/           # FastAPI routes
│   ├── services/      # Business logic
│   ├── models/        # Pydantic models
│   ├── utils/         # Helper functions
│   ├── config/        # Configuration
│   └── tests/         # Unit tests
├── weights/           # Model weights
└── data/             # Sample data
```

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- CUDA-capable GPU (optional, for faster inference)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd suture-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the sample environment file:
```bash
cp .env.example .env
```

5. Download the YOLOv8 model weights:
```bash
mkdir -p weights
# Download weights to weights/yolov8n.pt
```

## Running the Application

### Development Mode

```bash
uvicorn backend.app.api.main:app --reload
```

### Docker

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

## API Endpoints

- `GET /health`: Check API and model status
- `POST /analyze`: Analyze an image for sutures

## Testing

Run the test suite:
```bash
pytest backend/app/tests
```

## Environment Variables

See `.env.example` for all available configuration options.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
