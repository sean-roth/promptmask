# PromptMask Quick Start Guide

**Status:** Pre-Development  
**Ready to code:** Once SAM 3 access approved

## For Sean (Developer)

### Immediate Next Steps

1. **Request SAM 3 Access** (if not done)
   - Visit: https://huggingface.co/facebook/sam3
   - Click "Request Access"
   - Wait ~24 hours for approval

2. **Set Up Development Environment**
   ```bash
   # Clone repo
   git clone https://github.com/sean-roth/promptmask.git
   cd promptmask
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies (once created)
   pip install -r requirements.txt
   
   # Authenticate with Hugging Face
   huggingface-cli login
   ```

3. **Start Development**
   - Begin with basic Gradio app structure
   - Get SAM 3 model loading and running
   - Test with single image first
   - Then extend to video

### Development Priority Order

**Phase 1: Foundation (Days 1-2)**
- [ ] Create basic Gradio UI (upload + preview)
- [ ] Implement SAM 3 model loading
- [ ] Test inference on single image
- [ ] Verify MPS (Metal) acceleration working

**Phase 2: Video Pipeline (Days 3-4)**
- [ ] Video upload and validation
- [ ] Frame extraction with ffmpeg
- [ ] Process all frames with SAM 3
- [ ] Temporal tracking implementation

**Phase 3: Processing & Export (Days 5-6)**
- [ ] Temporal smoothing
- [ ] Edge refinement
- [ ] PNG sequence export
- [ ] Preview generation

**Phase 4: Integration (Days 7-8)**
- [ ] Feedback system
- [ ] Update checker
- [ ] Error handling
- [ ] Testing with friend

## For the Friend (User)

### When It's Ready

You'll receive a notification with:
1. **Installation Instructions**
   - Simple one-command install
   - Or downloadable .app file

2. **Quick Tutorial Video**
   - 5-minute walkthrough
   - Your specific workflow

3. **Test Video**
   - We'll process one of your sales videos together
   - You provide feedback
   - We iterate

### What to Prepare

**Have Ready:**
- 1-2 sample sales videos (MP4 format)
- Note what you'd normally mask in each
- Your typical export workflow documented

**Don't Need:**
- Any AI knowledge
- Command line experience
- Time investment (we'll make this dead simple)

## Technical Milestones

### Milestone 1: "Hello SAM 3"
- [ ] Model loads successfully
- [ ] Can segment single image with prompt
- [ ] Mask displays correctly
- **Est:** Day 2

### Milestone 2: "It Processes Video"
- [ ] Accepts video upload
- [ ] Processes all frames
- [ ] Shows progress
- [ ] Exports PNG sequence
- **Est:** Day 5

### Milestone 3: "Adobe Integration"
- [ ] Exports work in Premiere
- [ ] Exports work in After Effects
- [ ] Import guides included
- **Est:** Day 7

### Milestone 4: "Production Ready"
- [ ] Friend successfully uses it
- [ ] Saves 3+ hours on video
- [ ] Reports no major bugs
- [ ] Feedback system working
- **Est:** Day 10

## Resources

**Stitch UI Design:**  
https://stitch.withgoogle.com/projects/16389844715123694686

**SAM 3 Documentation:**
- GitHub: https://github.com/facebookresearch/sam3
- Hugging Face: https://huggingface.co/facebook/sam3
- Paper: (TBD)

**Gradio Examples:**
- https://gradio.app/docs
- https://github.com/gradio-app/gradio/tree/main/demo

**Similar Projects for Reference:**
- ComfyUI SAM nodes
- Roboflow SAM 3 integration

## Development Commands

```bash
# Run application
python app.py

# Run with debug mode
python app.py --debug

# Run tests
pytest

# Format code
black .

# Check types
mypy .

# Create distribution
pyinstaller app.py
```

## Notes for Claude Code

When working in this repo:

1. **Read SPEC.md first** - Understand the full vision
2. **Follow architecture** - Component structure matters
3. **Test incrementally** - Don't build everything at once
4. **Use type hints** - Makes debugging easier
5. **Log everything** - Debugging video processing is hard

## Communication Channels

**For Sean:**
- Issues: https://github.com/sean-roth/promptmask/issues
- Direct: Use feedback system once implemented

**For Friend:**
- In-app feedback button
- Text/call for urgent issues
- Screen share for complex problems

## Success Criteria

**We know it's working when:**
1. Friend processes a sales video in < 30 minutes (including setup)
2. Masks work in Premiere without manual adjustment
3. Friend says: "This is way better than doing it manually"
4. Zero data loss or corruption
5. No crashes during normal use

**We iterate until:**
- Processing time < 30 minutes for 3-minute video
- Mask accuracy > 90% on typical sales videos
- Export success rate > 95%
- Friend uses it for every video project

## Timeline

**Week 1: Core Development**
- Days 1-2: Foundation + SAM 3 integration
- Days 3-4: Video pipeline
- Days 5-6: Processing + export
- Day 7: Testing + fixes

**Week 2: Polish + Production**
- Day 8: Adobe integration
- Day 9: Feedback system
- Day 10: Friend testing
- Days 11-14: Iteration based on feedback

**Ready to ship:** Day 14 (2 weeks from start)

---

**Let's build this! 🚀**

Start by running:
```bash
huggingface-cli login
# Then request SAM 3 access
```