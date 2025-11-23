# PromptMask

> AI-powered video masking using natural language. Describe what you want to segment, get production-ready masks for Adobe workflows.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## What is PromptMask?

PromptMask eliminates hours of manual rotoscoping and masking work in video production. Instead of frame-by-frame manual work, simply describe what you want to segment:

- "person speaking"
- "product on desk" 
- "background trees"
- "speaker wearing blue shirt"

PromptMask uses Meta's SAM 3 (Segment Anything Model 3) to automatically generate accurate, temporally-consistent masks across your entire video.

## Who Is This For?

**Video Editors** working on:
- Sales and marketing videos
- Product demos
- Corporate communications
- Social media content
- Testimonials and interviews

**Use Cases:**
- Background replacement without green screen
- Selective color grading
- Object isolation for effects
- Privacy masking (blur faces, logos, screens)
- UI element highlighting in screen recordings

## Key Features

✅ **Text-Based Segmentation** - No manual clicking or drawing required
✅ **Temporal Consistency** - Stable masks that track objects across frames
✅ **Adobe Integration** - Export directly to Premiere Pro and After Effects
✅ **Local Processing** - Runs on your Mac, no cloud costs or data uploads
✅ **Quick Iteration** - Preview and refine before final export
✅ **Auto-Updates** - Integrated feedback system with one-click updates

## Quick Start

### Prerequisites

- macOS 12.0+ (Apple Silicon recommended)
- 16GB+ RAM
- Python 3.11+
- ~10GB free disk space (for model and temp files)

### Installation

```bash
# Clone the repository
git clone https://github.com/sean-roth/promptmask.git
cd promptmask

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Authenticate with Hugging Face (one-time setup)
huggingface-cli login

# Request access to SAM 3 model (one-time)
# Visit: https://huggingface.co/facebook/sam3
# Click "Request Access" and wait for approval (~24 hours)

# Launch the application
python app.py
```

The app will open automatically in your browser at `http://localhost:7860`

### First Video

1. **Upload** your video file (drag & drop or click to browse)
2. **Describe** what to segment: "person speaking"
3. **Click** "Process Video" 
4. **Wait** 15-30 minutes (processing time varies by video length)
5. **Preview** results with "Show Masks" toggle
6. **Export** in your preferred format for Adobe

## Export Formats

### After Effects
- JSON keyframe data with mask paths
- PNG sequence with alpha channel
- Import guide: [ADOBE_INTEGRATION.md](docs/ADOBE_INTEGRATION.md)

### Premiere Pro  
- PNG sequence as opacity mattes
- Automatically matches source resolution and framerate
- Import guide: [ADOBE_INTEGRATION.md](docs/ADOBE_INTEGRATION.md)

### Universal
- Black/white PNG masks (white = selected object)
- Works with any video editing software
- Frame-accurate, ready for compositing

## Performance

**On Apple M3 Pro (24GB RAM):**
- 1080p video: ~3-5 fps processing speed
- 3-minute video: ~18-30 minutes processing time
- Model loading: ~30 seconds (first run only)

**Tips for faster processing:**
- Lower resolution videos process faster
- Shorter clips for iteration, full resolution for final
- Close other applications during processing

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [User Workflow Guide](docs/USER_WORKFLOW.md)
- [Adobe Integration](docs/ADOBE_INTEGRATION.md)
- [Development Setup](docs/DEVELOPMENT.md)
- [Feedback System](docs/FEEDBACK_SYSTEM.md)

## Getting Help

Having issues? Use the feedback box in the app to report problems directly. Your feedback automatically creates a GitHub issue with all relevant context.

For immediate help:
- Check [Troubleshooting](docs/TROUBLESHOOTING.md)
- Review [Common Issues](docs/COMMON_ISSUES.md)
- Open an issue on GitHub

## Roadmap

**v0.1.0** (In Development)
- [ ] Basic video upload and processing
- [ ] Text prompt segmentation
- [ ] PNG sequence export
- [ ] After Effects JSON export
- [ ] Premiere Pro integration

**v0.2.0** (Planned)
- [ ] Multi-object tracking
- [ ] Frame-by-frame refinement
- [ ] Preset templates
- [ ] Batch processing

**v0.3.0** (Future)
- [ ] Cloud processing option
- [ ] Real-time preview
- [ ] Advanced mask refinement tools

## Technical Details

- **SAM 3**: 848M parameter model from Meta AI
- **Backend**: Python with PyTorch (Metal Performance Shaders)
- **Frontend**: Gradio web interface
- **Video Processing**: ffmpeg for frame extraction
- **Export**: Pillow for image processing, JSON for After Effects data

## Contributing

This project is designed for a specific workflow but pull requests are welcome for bug fixes and performance improvements.

## License

MIT License - see [LICENSE](LICENSE) for details

## Credits

- Built on [Meta's SAM 3](https://github.com/facebookresearch/sam3)
- UI design via [Google Stitch](https://stitch.withgoogle.com)
- Developed for professional video editing workflows

## Acknowledgments

Thanks to Meta AI Research for releasing SAM 3 as open source, making this tool possible.