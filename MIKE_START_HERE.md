# 🎬 PromptMask - Quick Start for Mike

**Welcome Mike!** This guide will get you up and running in ~15 minutes.

## What This Tool Does

Upload a video → Type what to segment (like "person" or "product") → Get masks for Adobe Premiere/After Effects

Replaces 3-4 hours of manual work with 30 minutes of automated processing.

---

## Setup (One-Time, ~10 minutes)

### Step 1: Download the Code

1. Go to: https://github.com/sean-roth/promptmask
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Unzip to your Desktop (creates folder: `promptmask-main`)

### Step 2: Install Python (If Not Already Installed)

**On Mac:**
1. Open Terminal (press Cmd+Space, type "Terminal")
2. Copy/paste this command:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
3. Wait for Homebrew to install
4. Then run:
```bash
brew install python@3.11
```

**On Windows:**
1. Go to: https://www.python.org/downloads/
2. Download Python 3.11
3. Run installer, **check "Add Python to PATH"**
4. Click Install

### Step 3: Install Dependencies

**On Mac:**
```bash
# Open Terminal
cd ~/Desktop/promptmask-main

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install packages (takes 5-10 minutes)
pip install -r requirements.txt
```

**On Windows:**
```bash
# Open Command Prompt
cd C:\Users\YourName\Desktop\promptmask-main

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install packages (takes 5-10 minutes)
pip install -r requirements.txt
```

### Step 4: Add HuggingFace Token

1. Create a file called `.env` in the `promptmask-main` folder
2. Open it with Notepad (Windows) or TextEdit (Mac)
3. Type this (use the token Sean gave you):
   ```
   HUGGINGFACE_TOKEN=hf_xxxxx
   ```
4. Save the file
5. That's it!

**What the token looks like:**
- Starts with `hf_`
- About 40 characters long
- Sean will text it to you

**Example .env file:**
```
HUGGINGFACE_TOKEN=hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890
```

---

## Running the App (Every Time)

### On Mac:
```bash
# Open Terminal
cd ~/Desktop/promptmask-main

# Activate virtual environment
source venv/bin/activate

# Run the app
python app.py
```

### On Windows:
```bash
# Open Command Prompt
cd C:\Users\YourName\Desktop\promptmask-main

# Activate virtual environment
venv\Scripts\activate

# Run the app
python app.py
```

**The app will open in your browser automatically at http://localhost:7860**

---

## Using the App

### First Time (Downloads Model - 10 minutes)

1. Click **"🚀 Load SAM 3 Model"**
2. Wait ~10 minutes for download (only happens once!)
3. You'll see "✅ Model loaded successfully"

### Every Other Time (Instant)

Model loads from cache in 30-60 seconds.

### Processing a Video

1. **Upload Video**: Drag & drop your video file (MP4, MOV, AVI)
2. **Enter Prompt**: Type what to segment:
   - "person" - segments all people
   - "person speaking" - segments the speaker
   - "product on table" - segments the product
   - "hand holding object" - segments hands with objects
3. **Select Preset**:
   - 👤 **Speaker Isolation** - For talking head videos
   - 📦 **Product Demo** - For product videos
   - ⚙️ **Custom** - Manual settings
4. **Click "🎬 Process Video"**
5. **Wait** - Takes about 30 minutes per minute of video
6. **Download** - Get your PNG sequence + preview

---

## What You'll Get

After processing:
- **Folder of PNG images** - One mask per frame
- **Preview image** - Shows sample results
- **ZIP file** - Contains everything

### Using in Adobe

**Premiere Pro:**
1. Import → Import Media
2. Select first PNG (e.g., `mask_0001.png`)
3. Check "Image Sequence"
4. Drag to timeline above your video
5. Set blend mode to use as mask

**After Effects:**
1. Import PNG sequence
2. Use as track matte or alpha channel
3. Apply effects to isolated subject

---

## Troubleshooting

### "Model not loaded"
- Click the "Load SAM 3 Model" button first
- Wait for it to say "Model loaded successfully"

### "Authentication failed" or "Missing HuggingFace token"
- Make sure you created the `.env` file
- Check that it contains: `HUGGINGFACE_TOKEN=hf_xxxxx`
- Make sure the `.env` file is in the `promptmask-main` folder (same folder as `app.py`)
- Ask Sean for the token if you don't have it

### "Out of memory"
- Close other applications
- Try a shorter video (under 30 seconds)
- Restart the app

### "Can't find python command"
**Mac:** Use `python3` instead of `python`
**Windows:** Make sure you checked "Add to PATH" during install

### App won't start
- Make sure virtual environment is activated
- You should see `(venv)` in your terminal prompt
- If not, run the activate command again

### Processing takes forever
- Normal! SAM 3 is processing each frame
- Expect ~30 minutes per minute of video on Mac
- ~15-20 minutes on Windows with good GPU

---

## Hardware Recommendations

### Minimum
- 16GB RAM
- Any recent Mac or Windows PC
- 20GB free disk space

### Recommended  
- 24GB+ RAM
- M3 Pro/Max Mac or NVIDIA GPU
- 50GB free disk space

### What to Expect

**M3 Pro Mac:**
- First run: 10 min download + processing
- Processing: ~30 min per minute of video
- Quality: Excellent

**Windows with NVIDIA GPU:**
- First run: 10 min download + processing
- Processing: ~15-20 min per minute of video
- Quality: Excellent

**CPU only (no GPU):**
- Processing: 2-3 hours per minute of video
- Not recommended for regular use

---

## Example Workflow

**Let's say you have a 30-second product demo video:**

1. Start app → Load model (30 seconds)
2. Upload video
3. Type prompt: "product on table"
4. Click "Product Demo" preset
5. Click "Process Video"
6. Wait ~15 minutes
7. Download PNG sequence
8. Import to Premiere/After Effects
9. Done! 🎉

**Manual rotoscoping would take 1.5-2 hours. This takes 15 minutes!**

---

## Need Help?

**Issues with setup?**
- Text Sean with screenshot of error
- Include what command you ran

**Issues with quality?**
- Try different text prompts
- Adjust confidence threshold (Advanced Options)
- Increase feather radius for softer edges

**Feature requests?**
- Tell Sean what you need
- He can add it in future updates

---

## Advanced Tips (Optional)

### Better Results

1. **Use specific prompts**: "person in blue shirt" vs just "person"
2. **Adjust confidence**: Lower = more aggressive masking
3. **Enable temporal smoothing**: Reduces flicker between frames
4. **Increase feather**: Makes edges softer/more natural

### Batch Processing

Want to process multiple videos? Just repeat the process:
1. Process video 1
2. Download results
3. Upload video 2
4. Process again

### Keyboard Shortcuts

None yet, but coming soon!

---

## Summary

✅ **Setup once** (~15 min)
✅ **Run app** (30 seconds after first time)
✅ **Process video** (~30 min per min of video)
✅ **Get Adobe-ready masks** instantly

**Questions?** Text Sean!

**Ready to start?** Go back to Step 1! 🚀
