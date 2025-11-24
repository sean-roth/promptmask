# PromptMask Installation Guide

This guide will help you install and set up PromptMask for local SAM 3 model inference.

## System Requirements

### Minimum Requirements
- **OS**: macOS 12.0+ (Apple Silicon recommended), Linux, or Windows
- **RAM**: 16GB minimum
- **Python**: 3.11 or higher
- **Disk Space**: ~10GB free (for model and temp files)

### Recommended Setup
- **Hardware**: Apple M3 Pro/Max with 24GB+ RAM, or NVIDIA GPU with CUDA support
- **Disk Space**: 20GB free for model + temporary files

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/sean-roth/promptmask.git
cd promptmask
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- PyTorch (with Metal/CUDA support)
- Transformers (for SAM 3 model)
- Gradio (for web interface)
- OpenCV and other processing libraries

## Hugging Face Authentication (Required)

SAM 3 requires authentication to download the model (gated access by Meta).

### One-Time Setup

1. **Create a Hugging Face account** at https://huggingface.co

2. **Request access to SAM 3** at https://huggingface.co/facebook/sam3
   - Click "Request Access"
   - Wait for approval (usually within 24 hours)

3. **Generate a read token** at https://huggingface.co/settings/tokens
   - Click "New token"
   - Select "Read" access
   - Copy the token

4. **Authenticate on your machine**:

```bash
huggingface-cli login
# Paste your token when prompted
```

### First Run

The first time you run PromptMask, it will download the SAM 3 model (~3GB) to:
- **Linux/Mac**: `~/.cache/huggingface/hub/`
- **Windows**: `C:\Users\<username>\.cache\huggingface\hub\`

This takes 5-10 minutes on a fast connection. Subsequent runs use the cached model and start instantly.

## Verifying Installation

Test the setup with this Python script:

```python
from transformers import Sam3Model, Sam3Processor
import torch

# Should download model on first run
model = Sam3Model.from_pretrained("facebook/sam3")
processor = Sam3Processor.from_pretrained("facebook/sam3")
print("✅ SAM 3 loaded successfully!")

# Check device
if torch.cuda.is_available():
    print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("✅ Using Apple Silicon GPU (MPS)")
else:
    print("ℹ️ Using CPU (no GPU available)")
```

## Running PromptMask

After installation and authentication:

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Launch the application
python app.py
```

The app will open automatically in your browser at `http://localhost:7860`

## Troubleshooting

### Authentication Errors

If you see authentication errors:

```
Failed to load SAM 3 model: 401 Unauthorized
```

**Solution:**
1. Make sure you've requested access at https://huggingface.co/facebook/sam3
2. Wait for approval (check your email)
3. Run `huggingface-cli login` again with a valid token

### CUDA/GPU Not Detected

If you have an NVIDIA GPU but PyTorch uses CPU:

**Solution:**
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Apple Silicon Issues

If MPS (Metal Performance Shaders) is not working:

**Solution:**
- Update to macOS 12.3 or later
- Update PyTorch: `pip install --upgrade torch torchvision`

### Out of Memory Errors

If you encounter memory errors during processing:

**Solution:**
1. Close other applications
2. Process shorter video clips
3. Lower the video resolution
4. Use CPU instead of GPU (slower but uses less memory)

### Model Download Fails

If the model download is interrupted or fails:

**Solution:**
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/hub/models--facebook--sam3

# Try downloading again
python -c "from transformers import Sam3Model; Sam3Model.from_pretrained('facebook/sam3')"
```

## Performance Notes

### Processing Speed
- **M3 Pro (24GB)**: 3-5 fps for 1080p video
- **NVIDIA RTX 3080**: 5-8 fps for 1080p video
- **CPU only**: 0.5-1 fps for 1080p video

### First Run
- Model download: 5-10 minutes (one-time)
- Model loading: 30-60 seconds
- Subsequent runs: Instant (model cached)

## Offline Usage

After the initial model download, PromptMask works completely offline:
- No internet connection required
- No API calls
- All processing happens locally on your machine

## Getting Help

If you encounter issues:
1. Check this troubleshooting section
2. Review error messages in the terminal
3. Use the feedback box in the app to report issues
4. Open an issue on GitHub with error details

## Next Steps

After successful installation:
1. Read the [Quick Start Guide](QUICKSTART.md)
2. Review [Architecture Documentation](docs/ARCHITECTURE.md)
3. Learn about [Adobe Integration](docs/ADOBE_INTEGRATION.md)

---

**Note**: This application processes video locally on your machine. No data is sent to external servers (except for the one-time model download from Hugging Face).
