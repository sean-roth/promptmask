# PromptMask Product Specification
**Version:** 0.1.0  
**Last Updated:** November 23, 2025  
**Status:** Pre-Development

## Executive Summary

PromptMask is a desktop application for macOS that uses Meta's SAM 3 AI model to automatically generate video masks using natural language prompts. It eliminates 3-4 hours of manual rotoscoping per video, making professional video effects accessible to editors working on sales videos, product demos, and corporate content.

**Target User:** Video editor at B2B SaaS company working on sales/marketing content  
**Core Value:** "Describe what you want → Get masks → Import to Adobe → Done in 30 minutes"

---

## Problem Statement

### Current Pain Points

1. **Manual Rotoscoping is Tedious**
   - 3-4 hours per video for frame-by-frame masking
   - Repetitive, soul-crushing work
   - Takes time away from creative decisions

2. **Existing Tools Are Too Complex**
   - After Effects rotobrush: still requires manual cleanup
   - Mocha tracking: steep learning curve
   - Cloud services: expensive, data leaves machine

3. **B2B Video Workflows Are Repetitive**
   - Same types of shots: speaker, product demo, testimonial
   - Same effects needed: background replacement, selective grading, UI highlighting
   - High volume: 10-20 videos per month

### Target User Profile

**Name:** Video Editor at B2B Company  
**Context:** 
- Works for AI market research SaaS company
- Uses Adobe Premiere Pro + After Effects
- 2024 MacBook Pro (Apple Silicon)
- Produces sales videos, product demos, enterprise content
- Processes videos one at a time (not batch)
- Doesn't want to learn "AI" - just wants better tools

**Pain:** Spends 3-4 hours manually masking speakers for background replacement and UI elements for highlighting

**Desired Outcome:** Spend 20 minutes setting up automated masks, focus remaining time on creative polish

---

## Solution Overview

### Core Concept

"ChatGPT for video masking" - you describe what you want segmented, SAM 3 finds it and tracks it across frames, you import the masks into your existing workflow.

### Key Innovation

**Text-based prompting** instead of manual clicking:
- Current: Click object in frame 1 → manually adjust in frames 2-500 → cry
- PromptMask: Type "person speaking" → get masks for all 500 frames → coffee break

### Unique Positioning

1. **Local-first**: Runs on your Mac, no cloud uploads
2. **Adobe-integrated**: Exports exactly what Premiere/AE expect
3. **Workflow-focused**: Designed around real editing workflows, not AI demos
4. **No AI knowledge required**: It's just a tool that makes masks

---

## User Experience

### Happy Path Workflow

```
1. FILM VIDEO (existing workflow)
   ↓
2. ROUGH CUT IN PREMIERE (existing workflow)
   ↓
3. OPEN PROMPTMASK
   - Drag in video file
   - Type: "person speaking"
   - Click "Process Video"
   - Wait 20 minutes (get coffee)
   ↓
4. PREVIEW RESULTS
   - Scrub timeline with mask overlay
   - Verify quality
   - Export for Premiere
   ↓
5. BACK TO PREMIERE
   - Import mask sequence
   - Apply effects using masks
   - Finish edit
   ↓
6. DONE IN 30 MIN vs 4 HOURS
```

### UI Design Philosophy

**Principles:**
1. **Calm, not flashy** - Professional tool, not consumer app
2. **Gradio aesthetic** - Clean, functional, minimal
3. **Apple-inspired** - Rounded corners, subtle shadows, system fonts
4. **Three-column layout** - Input | Preview | Export (clear flow)
5. **Progress visibility** - Always show what's happening and time remaining

**UI Reference:** https://stitch.withgoogle.com/projects/16389844715123694686

---

## Feature Specification

### MVP Features (v0.1.0)

#### 1. Video Upload & Validation

**Functionality:**
- Drag-and-drop or click to browse
- Supported formats: MP4, MOV, AVI
- Max file size: 5GB
- Max resolution: 4K
- Max duration: 30 minutes

**Validation:**
- Check format compatibility
- Extract metadata (resolution, fps, codec, duration)
- Generate thumbnail preview
- Show estimated processing time

**UI Elements:**
- Large drop zone with dashed border
- Thumbnail preview on upload
- Metadata display (1920x1080, 30fps, 3:24)
- Clear error messages for invalid files

#### 2. Text Prompt Segmentation

**Functionality:**
- Single text input field
- Natural language understanding via SAM 3
- Examples provided in placeholder text

**Supported Prompts:**
- Objects: "laptop", "coffee cup", "product"
- People: "person speaking", "speaker in blue shirt"
- UI elements: "search bar", "dashboard panel"
- Backgrounds: "everything except person", "office walls"

**Prompt Guidelines (shown in UI):**
- Be specific: "person in blue shirt" > "person"
- Include location: "person on left" (if multiple people)
- Include color: "red car" vs "car"

**UI Elements:**
- Text input with clear label
- Placeholder: "What should I segment? (e.g., 'person speaking')"
- Character counter (optional)
- Prompt history dropdown (future)

#### 3. Processing Settings

**Quality Slider:**
- Fast: Quicker, 85% accuracy
- Balanced: Default, 90% accuracy  
- Accurate: Slower, 95% accuracy

**Preset Buttons:**
- "Speaker Isolation" - Optimized for people
- "Product Demo" - Balanced settings
- "Custom" - Manual control

**Advanced Options (collapsed):**
- Edge feathering: 0-20px (default: 10px)
- Temporal smoothing: On/Off (default: On)
- Confidence threshold: 0-100% (default: 70%)

**UI Elements:**
- Slider with labels at each end
- Three prominent preset buttons
- Collapsible "Advanced" section
- Tooltips explaining each setting

#### 4. Video Processing

**SAM 3 Integration:**
- Model auto-downloads on first run (3.4GB)
- Loads into memory with MPS backend
- Processes frames sequentially
- Tracks objects temporally for consistency

**Processing Pipeline:**
1. Extract frames via ffmpeg → temp directory
2. Load SAM 3 model → GPU memory
3. Process first frame with prompt → initial mask
4. Propagate across remaining frames → tracked masks
5. Apply temporal smoothing → reduce flicker
6. Refine edges → better compositing
7. Generate preview video → with mask overlay
8. Ready for export

**Progress Indicators:**
```
Processing: Frame 245 / 1,200
█████████████░░░░░░░░ 20%
Est. time remaining: 24 minutes
Current confidence: 94%
```

**UI Elements:**
- Large progress bar
- Frame counter
- Time remaining estimate  
- Current confidence score
- Ability to cancel
- Log window (optional, for debugging)

#### 5. Preview System

**Functionality:**
- Video player with processed masks
- Toggle mask overlay on/off
- Scrub through timeline
- See confidence scores per frame

**Quality Indicators:**
- Green: High confidence (90%+)
- Yellow: Medium confidence (70-89%)
- Red: Low confidence (<70%)

**UI Elements:**
- Video player with standard controls
- Timeline scrubber
- "Show Masks" toggle button
- Frame number display
- Confidence meter

#### 6. Export System

**Format Options:**

**After Effects:**
- `masks_ae.json` - Keyframe data with bezier paths
- `masks/` - PNG sequence with alpha
- `import_guide_ae.txt` - Instructions

**Premiere Pro:**
- `masks/` - PNG sequence (opacity mattes)
- `metadata.xml` - Technical specs
- `import_guide_pr.txt` - Instructions

**Universal:**
- `masks/` - Standard black/white PNGs
- Works with any video editor

**Export UI:**
- Format dropdown (AE / Premiere / Universal)
- Output location selector
- "Export" button (enabled after processing)
- Progress indicator during export
- Success confirmation with file location

#### 7. Feedback System

**Functionality:**
- Text input for issue description
- Auto-collects system context
- Creates GitHub issue via API
- Returns issue number to user

**Context Collected:**
- User's description
- Video metadata (resolution, fps, duration)
- Processing settings (prompt, quality, etc.)
- Error logs
- App version
- System specs (OS, RAM, etc.)

**Privacy:**
- ✅ Video metadata
- ✅ System specs
- ✅ Error logs  
- ❌ NO video content
- ❌ NO personal info
- ❌ NO file paths

**UI Elements:**
- Text box at bottom of window
- Placeholder: "Having issues? Describe the problem..."
- "Send to Sean" button
- Confirmation message with issue number

#### 8. Update Notification

**Functionality:**
- Checks GitHub releases API on startup
- Compares current version to latest
- Shows notification badge if update available
- One-click update via git pull

**Update Process:**
1. User clicks "Update Now"
2. App runs: `git pull origin main`
3. App runs: `pip install -r requirements.txt`
4. App restarts automatically
5. User sees new version number

**UI Elements:**
- Green notification badge (🟢)
- "Update available: v0.3.3" message
- Brief description of what's fixed
- "Update Now" button
- Progress indicator during update

---

## Technical Specification

### Technology Stack

**Frontend:**
- **Gradio 4.x** - Web-based UI framework
- **Custom CSS** - Apple-inspired styling
- **HTML/JavaScript** - Interactive elements

**Backend:**
- **Python 3.11+** - Core language
- **PyTorch 2.3+** - ML framework
- **Metal Performance Shaders** - GPU acceleration on Apple Silicon
- **Meta SAM 3** - Segmentation model (848M parameters)

**Video Processing:**
- **ffmpeg** - Frame extraction, video manipulation
- **Pillow** - Image processing
- **OpenCV** - Computer vision utilities

**Export:**
- **JSON** - After Effects data format
- **PNG** - Mask image sequences
- **XML** - Metadata for Adobe

**Feedback:**
- **GitHub API** - Issue creation
- **requests** - HTTP client

### System Requirements

**Minimum:**
- macOS 12.0+
- 16GB RAM
- Python 3.11+
- 10GB free disk space

**Recommended:**
- macOS 14.0+
- Apple M3 Pro/Max with 24GB+ RAM
- 20GB free disk space for model + temp files

**Performance:**
- M3 Pro: 3-5 fps processing speed (1080p)
- M1/M2: 2-4 fps processing speed
- Intel Mac: Not recommended (CPU-only, very slow)

### File Structure

```
promptmask/
├── app.py                    # Main entry point
├── requirements.txt          # Dependencies
├── .env                      # Configuration
│
├── ui/                       # Gradio interface
│   ├── components.py
│   ├── themes.py
│   └── callbacks.py
│
├── pipeline/                 # Video processing
│   ├── video_processor.py
│   ├── validators.py
│   └── file_manager.py
│
├── sam3/                     # SAM 3 integration
│   ├── model_loader.py
│   ├── inference.py
│   └── video_tracker.py
│
├── processing/               # Post-processing
│   ├── mask_refiner.py
│   ├── temporal_smoother.py
│   └── quality_checker.py
│
├── export/                   # Export handlers
│   ├── after_effects.py
│   ├── premiere_pro.py
│   └── universal.py
│
├── feedback/                 # Feedback system
│   ├── collector.py
│   ├── github_integration.py
│   └── update_checker.py
│
└── tests/                    # Test suite
    ├── test_video_processor.py
    ├── test_sam3_inference.py
    └── test_exporters.py
```

### Data Flow

```
User → Gradio UI → Video Pipeline → SAM 3 Engine → Processing Layer → Export Layer → User

Parallel: Feedback System → GitHub Issues → Update Checker → Notification → User
```

### Security & Privacy

**Local Processing:**
- All computation happens on user's Mac
- No data sent to cloud (except model download)
- Temp files auto-deleted after processing

**Feedback System:**
- Only metadata collected (no video content)
- User reviews what's sent before submission
- Can be disabled in settings

---

## Development Roadmap

### Phase 1: MVP (v0.1.0) - 2 weeks

**Week 1:**
- [ ] Set up project structure
- [ ] Implement Gradio UI
- [ ] Video upload & validation
- [ ] SAM 3 model integration
- [ ] Basic inference pipeline

**Week 2:**
- [ ] Temporal smoothing
- [ ] Export system (PNG sequences)
- [ ] Feedback system
- [ ] Update notification
- [ ] Testing & debugging

**Deliverable:** Working application, export to PNG sequence only

### Phase 2: Adobe Integration (v0.2.0) - 1 week

- [ ] After Effects JSON export
- [ ] Premiere Pro XML metadata
- [ ] Import scripts/guides
- [ ] Test with real Adobe projects
- [ ] Documentation

**Deliverable:** Full Adobe workflow integration

### Phase 3: Polish & Features (v0.3.0) - 1 week

- [ ] Multi-object tracking
- [ ] Frame-by-frame refinement UI
- [ ] Preset templates library
- [ ] Performance optimizations
- [ ] Comprehensive error handling

**Deliverable:** Production-ready tool

### Phase 4: Advanced Features (v1.0.0) - Future

- [ ] Batch processing
- [ ] Cloud processing option
- [ ] Real-time preview
- [ ] Plugin system
- [ ] Community preset sharing

---

## Success Metrics

### User Success

**Primary Metric:** Time saved per video
- Target: 3+ hours saved per video
- Measurement: User reports, time tracking

**Secondary Metrics:**
- Mask quality: 90%+ confidence average
- Export success rate: 95%+ imports work in Adobe
- User satisfaction: NPS > 50

### Technical Success

**Performance:**
- Processing speed: 3-5 fps on M3 Pro (1080p)
- Memory usage: < 8GB during processing
- App startup time: < 5 seconds

**Reliability:**
- Crash rate: < 1% of sessions
- Export success: 98%+ valid outputs
- Update success: 100% successful updates

### Adoption Success

**Initial Target:** 1 user (the friend)
- If it works well, word-of-mouth to colleagues
- Potential internal tool at his company

**Future:** Open source project
- GitHub stars > 100 in first month
- Active issue resolution
- Community contributions

---

## Risk Assessment

### Technical Risks

**Risk:** SAM 3 model too slow on M1/M2 Macs  
**Mitigation:** Offer resolution downscaling, cloud processing option

**Risk:** Export formats incompatible with Adobe updates  
**Mitigation:** Version testing, community feedback, quick patches

**Risk:** Memory issues with long videos  
**Mitigation:** Frame streaming, smart garbage collection, temp file management

### User Experience Risks

**Risk:** Prompts don't work as expected  
**Mitigation:** Clear examples, prompt suggestions, iteration support

**Risk:** Processing time too long  
**Mitigation:** Clear time estimates, progress visibility, cancel option

**Risk:** Results not good enough for production use  
**Mitigation:** Quality presets, manual refinement tools, realistic expectations

### Business Risks

**Risk:** Friend's workflow changes (no longer needs tool)  
**Mitigation:** Flexible architecture, other potential users identified

**Risk:** SAM 3 license changes  
**Mitigation:** Model is open source under permissive license

---

## Open Questions

1. **Should we support Intel Macs?** 
   - Probably no - performance would be terrible
   - Focus on Apple Silicon only

2. **Cloud processing option from day 1?**
   - No - local-first for MVP
   - Can add in v0.3.0 if needed

3. **Batch processing in MVP?**
   - No - target user does one video at a time
   - Can add in v0.3.0 for power users

4. **Pricing model?**
   - Free and open source for now
   - Potential paid cloud processing later
   - Or freemium model with advanced features

5. **Windows/Linux support?**
   - Not in MVP
   - If demand exists, can add later
   - PyTorch supports both, mainly UI testing needed

---

## Next Steps

1. **This conversation:** Finalize spec, create repo ✅
2. **UI Design:** Refine Stitch mockup → HTML/CSS ✅
3. **Development:** Start with basic Gradio app
4. **SAM 3 Integration:** Get model running locally
5. **Feedback Loop:** Friend tests, reports issues, we iterate

**Ready to start coding!** 🚀