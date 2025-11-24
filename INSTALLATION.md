# PromptMask Installation Guide

## System Requirements

### Minimum Requirements
- **macOS**: 12.0+ (Monterey or later)
- **RAM**: 16GB minimum
- **Storage**: 10GB free space (for model and temporary files)
- **Python**: 3.11 or higher
- **Apple Silicon**: M1/M2/M3 recommended (for MPS acceleration)

### Recommended Setup
- **macOS**: 14.0+ (Sonoma)
- **Processor**: Apple M3 Pro/Max with 24GB+ RAM
- **Storage**: 20GB+ free space
- **Python**: 3.11+

## Installation Steps

### 1. Install System Dependencies

#### Install Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Install Python 3.11+
```bash
brew install python@3.11
```

#### Install ffmpeg (required for video processing)
```bash
brew install ffmpeg
```

### 2. Clone the Repository

```bash
git clone https://github.com/sean-roth/promptmask.git
cd promptmask
```

### 3. Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**Note**: You'll need to activate the virtual environment every time you run PromptMask:
```bash
source venv/bin/activate
```

### 4. Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

This will install:
- Gradio (UI framework)
- PyTorch with MPS support (for Apple Silicon GPU acceleration)
- Hugging Face Transformers (for SAM 3 model)
- OpenCV, Pillow, ffmpeg-python (video/image processing)
- SciPy, NumPy (scientific computing)

### 5. Authenticate with Hugging Face

PromptMask uses Meta's SAM 3 model from Hugging Face. You need to:

#### a. Create Hugging Face Account
1. Go to https://huggingface.co/join
2. Create a free account

#### b. Request Access to SAM 3
1. Visit https://huggingface.co/facebook/sam3
2. Click "Request Access"
3. Wait for approval (usually ~24 hours)

#### c. Get Access Token
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "promptmask")
4. Select "read" access
5. Click "Generate token"
6. Copy the token

#### d. Login via CLI
```bash
huggingface-cli login
```

Paste your token when prompted.

### 6. Verify Installation

```bash
# Check Python version
python --version
# Should show Python 3.11.x or higher

# Check ffmpeg
ffmpeg -version
# Should show ffmpeg version info

# Check PyTorch MPS support (on Apple Silicon)
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
# Should show: MPS available: True (on Apple Silicon)
```

## First Run

### Launch PromptMask

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Launch the application
python app.py
```

On first run:
1. The SAM 3 model will download (~3.4GB) - this takes 5-10 minutes
2. Model will load into memory - takes ~30-60 seconds
3. Gradio interface will open in your browser at http://localhost:7860

### Test with a Sample Video

1. Drag and drop a short video file (MP4, MOV, or AVI)
2. Enter a prompt: "person speaking"
3. Click "Process Video"
4. Wait for processing to complete
5. Preview the result with mask overlay
6. Export as PNG sequence

## Troubleshooting

### Issue: "No module named 'sam3'"

**Solution**: Make sure you're in the correct directory and virtual environment:
```bash
cd /path/to/promptmask
source venv/bin/activate
python app.py
```

### Issue: "Failed to load model"

**Possible causes**:
1. No Hugging Face authentication
2. No access to SAM 3 model
3. No internet connection during first run

**Solution**:
```bash
# Check if you're logged in
huggingface-cli whoami

# If not, login again
huggingface-cli login

# Verify SAM 3 access at https://huggingface.co/facebook/sam3
```

### Issue: "MPS not available" or slow performance

**Solution**:
- MPS (Metal Performance Shaders) is only available on Apple Silicon (M1/M2/M3)
- On Intel Macs, the app will fall back to CPU (very slow)
- Consider using a machine with Apple Silicon for best performance

### Issue: "ffmpeg: command not found"

**Solution**:
```bash
# Install ffmpeg
brew install ffmpeg

# Verify installation
ffmpeg -version
```

### Issue: Out of memory during processing

**Solutions**:
1. Use "fast" quality setting (processes at lower resolution)
2. Close other applications
3. Process shorter videos first
4. Upgrade to machine with more RAM

### Issue: Video processing takes too long

**Expected processing times** (on M3 Pro, 1080p video):
- 1-minute video: ~6-10 minutes
- 3-minute video: ~18-30 minutes
- 5-minute video: ~30-50 minutes

**To speed up**:
1. Use "fast" quality mode
2. Disable temporal smoothing
3. Process at lower resolution
4. Use shorter clips during iteration

## Updating PromptMask

```bash
cd promptmask
git pull origin main
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

## Uninstallation

```bash
# Remove virtual environment
rm -rf venv

# Remove repository
cd ..
rm -rf promptmask

# Optionally, remove cached models
rm -rf ~/.cache/huggingface/hub/models--facebook--sam3
```

## Getting Help

- **Documentation**: https://github.com/sean-roth/promptmask
- **Issues**: https://github.com/sean-roth/promptmask/issues
- **In-App Feedback**: Use the feedback button (coming in v0.2.0)

## Next Steps

Once installed, read:
1. [User Workflow Guide](docs/USER_WORKFLOW.md) - How to use PromptMask
2. [Adobe Integration](docs/ADOBE_INTEGRATION.md) - Import masks into Premiere/After Effects
3. [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions
