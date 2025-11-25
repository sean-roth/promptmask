# 🎬 PromptMask - Windows Setup Guide

**Welcome!** This guide will get you up and running on Windows in ~15 minutes.

## What This Tool Does

Upload a video → Type what to segment (like "person" or "product") → Get masks for Adobe Premiere/After Effects

Replaces 3-4 hours of manual work with 30 minutes of automated processing.

---

## Setup (One-Time, ~15 minutes)

### Step 1: Download the Code

1. Go to: https://github.com/sean-roth/promptmask
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Unzip to your Desktop (creates folder: `promptmask-main`)

### Step 2: Install Python (If Not Already Installed)

1. Go to: https://www.python.org/downloads/
2. Download **Python 3.11** (click the big yellow button)
3. Run the installer
4. **⚠️ IMPORTANT:** Check the box that says **"Add Python to PATH"**
5. Click "Install Now"

To verify it worked, open Command Prompt and type:
```bash
python --version
```
Should show something like `Python 3.11.x`

### Step 3: Install Dependencies

1. Open **Command Prompt** (press Windows key, type "cmd", hit Enter)
2. Navigate to the folder:
```bash
cd %USERPROFILE%\Desktop\promptmask-main
```

3. Create virtual environment:
```bash
python -m venv venv
```

4. Activate it:
```bash
venv\Scripts\activate
```

You should see `(venv)` at the start of your command line.

5. Install packages (takes 10-15 minutes first time):
```bash
pip install -r requirements.txt
```

### Step 4: Add HuggingFace Token

**Create the .env file:**

**Option A: Using Notepad**
1. Open Notepad
2. Type this EXACT line:
```
HUGGINGFACE_TOKEN=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
```
3. Click File → Save As
4. Navigate to your `promptmask-main` folder
5. In "File name" type: `.env` (include the dot!)
6. In "Save as type" select: **All Files**
7. Click Save

**Option B: Using Command Prompt**
```bash
cd %USERPROFILE%\Desktop\promptmask-main
echo HUGGINGFACE_TOKEN=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 > .env
```

**Verify it worked:**
```bash
type .env
```
Should show: `HUGGINGFACE_TOKEN=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`

That's it! You're ready to run the app.

---

## Running the App (Every Time)

```bash
# Open Command Prompt
cd %USERPROFILE%\Desktop\promptmask-main

# Activate virtual environment
venv\Scripts\activate

# Run the app
python app.py
```

**The app will open in your browser at http://localhost:7860**

(If it doesn't open automatically, just paste that URL in your browser)

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
5. **Wait** - Takes about 15-30 minutes per minute of video
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

### "python is not recognized"
- Python wasn't added to PATH during install
- Reinstall Python and make sure to check **"Add Python to PATH"**
- Or use the full path: `C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe`

### "Missing HuggingFace token"
- Make sure you created the `.env` file (not `.env.txt`)
- Check that it contains the EXACT line from Step 4
- The `.env` file should be in the `promptmask-main` folder

### Can't create .env file (saves as .env.txt)
- In Notepad's Save As dialog, change "Save as type" to **All Files**
- Or use the Command Prompt method in Step 4, Option B

### "Authentication failed" or "401 Unauthorized"
- Make sure the token line is copied EXACTLY (no spaces, no changes)
- Try deleting and recreating the `.env` file
- Text Sean if still having issues

### "Model not loaded"
- Click the "Load SAM 3 Model" button first
- Wait for it to say "Model loaded successfully"

### "ImportError: cannot import Sam3Model"
- Run: `pip install -r requirements.txt` again
- This reinstalls transformers from the latest source

### "venv\Scripts\activate" doesn't work
- Make sure you're in the right folder: `cd %USERPROFILE%\Desktop\promptmask-main`
- Try: `.\venv\Scripts\activate` (with the dot-backslash)

### "Out of memory"
- Close other applications
- Try a shorter video (under 30 seconds)
- Restart the app

### App won't start
- Make sure virtual environment is activated (you see `(venv)` in your prompt)
- If not, run: `venv\Scripts\activate`

### Processing is very slow
- Without an NVIDIA GPU, processing uses CPU (slower but works)
- Expect ~1-2 hours per minute of video on CPU
- With NVIDIA GPU: ~15-20 minutes per minute of video

---

## Hardware Info

### With NVIDIA GPU (Recommended)
- Processing: ~15-20 min per minute of video
- The app auto-detects and uses CUDA

### Without GPU (CPU only)
- Processing: ~1-2 hours per minute of video
- Still works, just slower
- Good for testing with short clips

### Minimum Requirements
- 16GB RAM
- 20GB free disk space
- Windows 10 or 11

---

## Quick Reference

**Start the app:**
```bash
cd %USERPROFILE%\Desktop\promptmask-main
venv\Scripts\activate
python app.py
```

**Check your .env file:**
```bash
type .env
```

**Reinstall dependencies:**
```bash
pip install -r requirements.txt
```

---

## Example Workflow

**Let's say you have a 30-second product demo video:**

1. Start app → Load model (30 seconds after first time)
2. Upload video
3. Type prompt: "product on table"
4. Click "Product Demo" preset
5. Click "Process Video"
6. Wait ~10-15 minutes (with GPU) or ~30-60 minutes (CPU)
7. Download PNG sequence
8. Import to Premiere/After Effects
9. Done! 🎉

---

## Need Help?

**Issues with setup?**
- Text Sean with screenshot of error
- Include what command you ran

**Issues with quality?**
- Try different text prompts
- Adjust confidence threshold (Advanced Options)
- Increase feather radius for softer edges

**Questions?** Text Sean!
