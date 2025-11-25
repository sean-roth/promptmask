---
title: PromptMask
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
license: mit
hardware: a10g-small
---

# 🎬 PromptMask - Video Segmentation with SAM 3

AI-powered video masking tool using Meta's SAM 3 model. Upload a video, describe what to segment with natural language prompts, and export production-ready masks for Adobe Premiere Pro and After Effects.

## Features

- **Natural Language Prompts**: Just describe what you want to segment ("person speaking", "product on table")
- **SAM 3 Integration**: Uses Meta's latest Segment Anything Model for state-of-the-art segmentation
- **Video Processing**: Handles full video sequences with temporal smoothing
- **Adobe-Ready Export**: PNG sequences ready for Premiere Pro and After Effects
- **Quality Presets**: Optimized settings for common use cases

## Usage

1. Click "Load SAM 3 Model" (first time downloads ~2GB, cached after)
2. Upload your video (MP4, MOV, AVI supported)
3. Enter a text prompt describing what to segment
4. Select a preset or adjust advanced options
5. Click "Process Video"
6. Download your mask sequence

## Example Prompts

- "person" - Segment all people
- "person speaking" - Segment the main speaker
- "product on table" - Segment a product demo
- "hand holding object" - Segment hands with objects
- "yellow school bus" - SAM 3 understands descriptive phrases

## Local Installation

See [MIKE_START_HERE.md](MIKE_START_HERE.md) for Mac or [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for Windows.

## Technical Details

- **Model**: facebook/sam3 via HuggingFace Transformers
- **Backend**: Gradio
- **Processing**: Temporal smoothing, mask refinement, feathering
- **Export**: Universal PNG with alpha channel

## License

MIT License - See [LICENSE](LICENSE) for details.
