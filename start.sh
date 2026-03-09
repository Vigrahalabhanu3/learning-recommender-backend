#!/usr/bin/env bash
set -e

echo "🚀 Starting Learning Recommender Backend..."

# Step 1: Download models if missing
echo "📦 Checking model files..."
python3 download_models.py

# Step 2: Start production server
echo "🌐 Starting Gunicorn server..."
exec gunicorn api:app --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120
