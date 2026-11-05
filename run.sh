#!/usr/bin/env bash
# AI Data Studio Launcher
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Check Python environment
if [ -d "/home/zarko/.gemini/antigravity/scratch/bitsy-ai-studio/.venv" ]; then
    PYTHON="/home/zarko/.gemini/antigravity/scratch/bitsy-ai-studio/.venv/bin/python"
elif [ -d "$DIR/.venv" ]; then
    PYTHON="$DIR/.venv/bin/python"
else
    PYTHON="$(command -v python3)"
fi

echo "=========================================================="
echo "⚡ Starting TechcodeX AI Data Studio..."
echo "📂 Project Directory: $DIR"
echo "🐍 Python Interpreter: $PYTHON"
echo "🌐 URL: http://127.0.0.1:7865"
echo "=========================================================="

exec "$PYTHON" app.py
