# Learning Recommender — Backend

Flask API powered by DistilBERT + Sentence Transformers + FAISS for personalized course recommendations.

## Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ⚠️ Model Files (Not in GitHub)

The model weight files are too large for GitHub. Download them and place them in the correct folders:

- `distilbert_finetuned/model.safetensors` (~255MB)
- `finetuned_recommender_v2/model.safetensors` (~418MB)

> Store these in **Google Drive** or **HuggingFace Hub** and download before running.

## Run Locally

```bash
source venv/bin/activate
python3 api.py
```

Server runs at: `http://localhost:5000`

## Run in Production

```bash
gunicorn api:app
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/recommend` | Get course recommendations |
| GET | `/health` | Health check |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PORT` | Server port (default: 5000) |
