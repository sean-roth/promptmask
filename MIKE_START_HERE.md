# 🎬 PromptMask - Quick Start for Mike

**Welcome Mike!** This guide will get you up and running in ~15 minutes.

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

### Step 3: Install Dependencies

**On Mac:**
```bash
# Open Terminal
cd ~/Desktop/promptmask-main

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install packages (takes 10-15 minutes first time)
pip install -r requirements.txt
```

### Step 4: Add HuggingFace Token

**Create the .env file:**

**On Mac:**
```bash
# In the promptmask-main folder, create .env file
nano .env

# Copy and paste this EXACT line:
HUGGINGFACE_TOKEN=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

# Press Ctrl+X, then Y, then Enter to save
```

**Important:**
- Copy the line EXACTLY as shown above
- Don't add spaces or extra lines
- Make sure it's called `.env` not `.env.txt`

That's it! You're ready to run the app.

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

### "Missing HuggingFace token"
- Make sure you created the `.env` file
- Check that it contains the EXACT line from Step 4
- The `.env` file should be in the `promptmask-main` folder (same place as `app.py`)

### "Authentication failed" or "401 Unauthorized"
- Copy the token line EXACTLY from Step 4 (no spaces, no changes)
- Make sure the file is named `.env` not `.env.txt`
- Try deleting and recreating the `.env` file
- Text Sean if still having issues

### "Model not loaded"
- Click the "Load SAM 3 Model" button first
- Wait for it to say "Model loaded successfully"
- If it says authentication failed, check your `.env` file

### "ImportError: cannot import Sam3Model"
- This means transformers needs to be reinstalled
- Run: `pip install -r requirements.txt` again
- This installs transformers from the latest source code

### Can't find .env file on Mac
- Mac hides files starting with `.` (dot)
- In Terminal, run: `ls -la` to see hidden files
- Or create it using: `nano .env` in Terminal

### Can't create .env file
- In Terminal, navigate to the folder: `cd ~/Desktop/promptmask-main`
- Create the file: `nano .env`
- Paste the token line
- Press Ctrl+X, then Y, then Enter

### "Out of memory"
- Close other applications
- Try a shorter video (under 30 seconds)
- Restart the app

### "Can't find python command"
Use `python3` instead of `python`

### App won't start
- Make sure virtual environment is activated
- You should see `(venv)` in your terminal prompt
- If not, run: `source venv/bin/activate`

### Processing takes forever
- Normal! SAM 3 is processing each frame
- Expect ~30 minutes per minute of video on Mac
- ~15-20 minutes on M3 Pro/Max

---

## Hardware Recommendations

### Minimum
- 16GB RAM
- Any M1/M2/M3 Mac
- 20GB free disk space

### Recommended  
- 24GB+ RAM
- M3 Pro/Max Mac
- 50GB free disk space

### What to Expect

**M3 Pro/Max Mac:**
- First run: 10 min download + processing
- Processing: ~20-30 min per minute of video
- Quality: Excellent

**M3 Base Mac:**
- Processing: ~30-40 min per minute of video
- Quality: Excellent (just slower)

---

## Example Workflow

**Let's say you have a 30-second product demo video:**

1. Start app → Load model (30 seconds after first time)
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

**Issues with .env file?**
- Text Sean with a screenshot
- Make sure you're in the right folder

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

### Checking Your Token

**On Mac:**
```bash
cat .env
```

Should show: `HUGGINGFACE_TOKEN=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`

---

## Summary

✅ **Setup once** (~15 min)
✅ **Create .env file** with token (copy from Step 4)
✅ **Run app** (30 seconds after first time)
✅ **Process video** (~30 min per min of video)
✅ **Get Adobe-ready masks** instantly

**Questions?** Text Sean!

**Ready to start?** Go back to Step 1! 🚀
