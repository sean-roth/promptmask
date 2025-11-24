# Installation Guide

Complete installation and setup guide for PromptMask.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- Git
- (Optional) CUDA-capable GPU for faster processing

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/sean-roth/promptmask.git
cd promptmask
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### GPU Support (Optional but Recommended)

For CUDA GPU support (much faster processing):

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For Apple Silicon (M1/M2/M3) MPS support:

```bash
# MPS is automatically available with PyTorch 2.0+ on macOS
pip install torch torchvision
```

### 4. HuggingFace Authentication

The SAM 3 model requires HuggingFace authentication to download.

#### Option A: Using huggingface-cli (Recommended)

```bash
# Install HuggingFace CLI (included in requirements.txt)
pip install huggingface-hub

# Login with your HuggingFace token
huggingface-cli login
```

When prompted:
1. Enter your HuggingFace token
   - Get your token at: https://huggingface.co/settings/tokens
   - Create a token with "Read" permission
2. Choose whether to add token to git credentials (optional)

#### Option B: Using Environment Variable

```bash
# Linux/Mac
export HUGGING_FACE_HUB_TOKEN="your_token_here"

# Windows (PowerShell)
$env:HUGGING_FACE_HUB_TOKEN="your_token_here"

# Windows (CMD)
set HUGGING_FACE_HUB_TOKEN=your_token_here
```

#### Option C: Using Python

Create a `.env` file in the project root:

```
HUGGING_FACE_HUB_TOKEN=your_token_here
```

### 5. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA: True  # or False if CPU only
```

## Running the Application

### Start the Gradio Interface

```bash
python app.py
```

The application will:
1. Start a local web server (default: http://localhost:7860)
2. Open automatically in your default browser
3. Display the model loading status

### First Run

On first run, the application will:
1. Download the SAM 3 model (~2GB)
2. Cache it locally in `~/.cache/huggingface/`
3. Load the model into memory

**Note:** First run may take 5-10 minutes depending on your internet speed.

## Usage

1. **Load Model**: Click "🚀 Load SAM 3 Model" button
   - Wait for "✅ Model loaded successfully" message
   - Model loads to GPU (CUDA/MPS) if available, otherwise CPU

2. **Upload Video**: Choose a video file (MP4, AVI, MOV, etc.)
   - Maximum file size: 500MB
   - Maximum frames: 300 (adjust in `VideoProcessor`)

3. **Enter Prompt**: Describe what to segment
   - Example: "person speaking"
   - Example: "product on table"

4. **Select Preset** (or customize):
   - 👤 Speaker Isolation: High confidence, strong feathering
   - 📦 Product Demo: Balanced settings
   - ⚙️ Custom: Manual control

5. **Process Video**: Click "🎬 Process Video"
   - Progress bar shows current status
   - Results appear in the right panel

6. **Download Results**:
   - ZIP file contains all mask frames
   - Preview shows sample frames

## Troubleshooting

### Authentication Errors

**Error:** "401 Client Error: Unauthorized"

**Solution:**
```bash
# Re-login to HuggingFace
huggingface-cli logout
huggingface-cli login
```

### CUDA Out of Memory

**Error:** "CUDA out of memory"

**Solutions:**
1. Reduce video resolution
2. Process fewer frames (adjust `max_frames`)
3. Use CPU instead (automatic fallback)

### Model Download Fails

**Error:** "Connection timeout" or "Download failed"

**Solutions:**
1. Check internet connection
2. Try again (download resumes automatically)
3. Manually download from HuggingFace:
   ```bash
   huggingface-cli download facebook/sam-3-large
   ```

### Import Errors

**Error:** "No module named 'X'"

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Video Format Not Supported

**Error:** "Unsupported video format"

**Solution:**
- Convert video to MP4 using ffmpeg:
  ```bash
  ffmpeg -i input.mov -c:v libx264 output.mp4
  ```

## System Requirements

### Minimum Requirements

- CPU: 4+ cores
- RAM: 8GB
- Storage: 5GB free space
- OS: Windows 10+, macOS 10.15+, or Linux

### Recommended Requirements

- CPU: 8+ cores
- RAM: 16GB
- GPU: NVIDIA RTX 3060 or better (6GB+ VRAM)
- Storage: 10GB free space

### Performance Benchmarks

Processing time for 100 frames at 720p:

| Hardware | Time |
|----------|------|
| CPU (Intel i7) | ~10 min |
| GPU (RTX 3060) | ~2 min |
| GPU (RTX 4090) | ~30 sec |
| Apple M2 Pro (MPS) | ~4 min |

## Advanced Configuration

### Adjust Frame Limit

Edit `app.py`:

```python
self.video_processor = VideoProcessor(max_frames=600)  # Default: 300
```

### Change Model

Edit `sam3/model_loader.py`:

```python
MODEL_ID = "facebook/sam-3-base"  # Smaller, faster model
```

Available models:
- `facebook/sam-3-base` (smaller, faster)
- `facebook/sam-3-large` (default, best quality)
- `facebook/sam-3-huge` (largest, highest quality)

### Custom Port

```bash
# Edit app.py, change:
interface.launch(server_port=8080)  # Default: 7860
```

## Offline Usage

After initial setup, the application can run offline:

1. Model is cached in `~/.cache/huggingface/`
2. No internet required after first download
3. All processing happens locally

## Updating

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstallation

```bash
# Remove virtual environment
deactivate
rm -rf venv/

# Remove cached models (optional)
rm -rf ~/.cache/huggingface/

# Remove project
cd ..
rm -rf promptmask/
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/sean-roth/promptmask/issues
- Documentation: See README.md and SPEC.md

## License

See LICENSE file for details.
