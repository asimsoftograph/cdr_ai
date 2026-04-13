# cdr-test

A lightweight OCR/ML API scaffold based on the provided project structure.

## Run locally

1. Activate your virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the API:
   ```bash
  uvicorn app.main:app --reload
   ```

The API will be available at `http://127.0.0.1:8000`.

## Project structure

- `app/` - FastAPI app, routes, ML pipeline, and utilities
- `models/` - model assets and weights
- `data/` - runtime logs and temporary files
- `tests/` - test placeholders
- `Dockerfile` - production container build
- `requirements.txt` - dependencies
