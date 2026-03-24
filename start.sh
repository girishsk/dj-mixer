#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== DJ Mixer Setup ==="
echo "Installing Python dependencies…"
pip install -r requirements.txt

echo ""
echo "=== Starting DJ Mixer ==="
echo "Open http://localhost:5000 in your browser"
echo ""
python app.py
