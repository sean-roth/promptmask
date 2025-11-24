#!/bin/bash
# PromptMask startup script

echo "🚀 Starting PromptMask..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found."
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import gradio" 2>/dev/null; then
    echo "❌ Dependencies not installed."
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

# Check ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  Warning: ffmpeg not found. Video processing will fail."
    echo "Install with: brew install ffmpeg"
fi

# Check Hugging Face login
if ! huggingface-cli whoami &> /dev/null; then
    echo "⚠️  Warning: Not logged in to Hugging Face."
    echo "Please run: huggingface-cli login"
fi

echo ""
echo "✅ Launching PromptMask..."
echo "   Interface will open at: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop the application."
echo ""

# Run the app
python app.py
