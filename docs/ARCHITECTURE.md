# PromptMask Architecture

Comprehensive technical architecture and design decisions for PromptMask.

## System Overview

See full architecture documentation at: https://github.com/sean-roth/promptmask

This document details:
- Component architecture
- Data flow
- SAM 3 integration
- Export systems
- Performance optimization
- Security considerations

## Quick Architecture Summary

**Key Components:**
1. **Gradio Frontend** - Web-based UI
2. **Video Pipeline** - ffmpeg-based processing
3. **SAM 3 Engine** - PyTorch + MPS
4. **Processing Layer** - Temporal smoothing & refinement
5. **Export Layer** - Adobe format generation
6. **Feedback System** - GitHub integration

**Technology Stack:**
- Python 3.11+
- PyTorch with Metal Performance Shaders
- Gradio 4.x
- ffmpeg for video
- Meta SAM 3 (848M parameters)

Full detailed architecture coming in v0.1.0 release.