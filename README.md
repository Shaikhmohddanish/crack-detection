# Crack Detection API

A Flask API that uses YOLO for crack detection in images.

## Features

- Upload images and get annotated results with detected cracks
- User-friendly web interface
- JSON endpoint for programmatic access
- Configurable confidence threshold and image size

## Deployment

This application is ready for deployment on Render:

1. Connect your GitHub repository to Render
2. Set up a new Web Service
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Use the Free instance type

## Local Development

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The application will be available at http://localhost:8000/# crack-detection
