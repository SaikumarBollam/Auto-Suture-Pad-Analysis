# Auto-Suture-Pad Analysis

A computer vision-based system for analyzing surgical sutures using YOLOv8 and OpenCV.

## Features

- Advanced suture detection using YOLOv8
- Real-time analysis capabilities
- Multiple preprocessing options
- Containerized deployment
- Model experiment tracking with MLflow
- Efficient model storage with MinIO
- Performance optimization with Redis caching

## Project Structure

```
├── config/               # Configuration files
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── src/                 # Source code
│   ├── api/            # FastAPI backend
│   ├── frontend/       # User interface
│   └── ml/            # Machine learning components
└── tests/              # Test suite
```

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- CUDA-capable GPU (optional)

## Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd auto-suture-pad
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the services:
   ```bash
   docker-compose up -d
   ```

The API will be available at http://localhost:8000

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
black src/ tests/
pylint src/ tests/
```

## API Endpoints

- `GET /health`: Health check
- `POST /analyze`: Analyze suture image
- Documentation available at `/docs` when running

## Configuration

Edit `config/config.yaml` to customize:
- Model parameters
- Preprocessing settings
- Storage configuration
- API settings

## License

MIT License - see LICENSE file for details