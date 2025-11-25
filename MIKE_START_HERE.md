# 🎬 PromptMask - Quick Start for Mike

**Welcome Mike!** This guide will get you up and running in ~20 minutes.

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

### Step 3: Request Access to SAM 3 Model

**⚠️ IMPORTANT: Do this BEFORE installing!**

SAM 3 is a "gated" model - you need to request access:

1. Go to: https://huggingface.co/facebook/sam3
2. Click **"Access repository"** (you'll need a free HuggingFace account)
3. Accept the terms
4. Access is usually approved instantly!

### Step 4: Get Your HuggingFace Token

1. Go to: https://huggingface.co/settings/tokens
2. Click **"Create new token"**
3. Name it anything (e.g., "PromptMask")
4. Set type to **"Read"**
5. Click **"Create"**
6. **Copy the token** (starts with `hf_...`)
7. Keep this page open - you'll need the token in Step 6

### Step 5: Install Dependencies

**On Mac:**
```bash
# Open Terminal
cd ~/Desktop/promptmask-main

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install packages (takes 10-15 minutes - SAM 3 needs latest transformers)
pip install -r requirements.txt
```

### Step 6: Add Your HuggingFace Token

**Create the .env file:**

**On Mac:**
```bash
# In the promptmask-main folder, create .env file
nano .env

# Paste this line, replacing YOUR_TOKEN_HERE with your actual token:
HUGGINGFACE_TOKEN=YOUR_TOKEN_HERE

# Press Ctrl+X, then Y, then Enter to save
```

Example (your token will be different):
```
HUGGINGFACE_TOKEN=hf_abcdefghijklmnopqrstuvwxyz123456
```

**Important:**
- Use YOUR OWN token from Step 4
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

### "Access denied" or "gated model" error
- Go to https://huggingface.co/facebook/sam3
- Click "Access repository" and accept terms
- Wait a few seconds and try again

### "Authentication failed" or "401 Unauthorized"
- Make sure you got YOUR OWN token from https://huggingface.co/settings/tokens
- The token should start with `hf_`
- Check your .env file has the token with no extra spaces
- Try creating a new token if the old one doesn't work

### "Model not loaded"
- Click the "Load SAM 3 Model" button first
- Wait for it to say "Model loaded successfully"
- If it says authentication failed, check your token

### "ImportError: cannot import Sam3Model"
- The transformers library needs to be installed from source
- Run: `pip install git+https://github.com/huggingface/transformers.git`
- Or just run `pip install -r requirements.txt` again

### Can't find .env file on Mac
- Mac hides files starting with `.` (dot)
- In Terminal, run: `ls -la` to see hidden files
- Or create it using: `nano .env` in Terminal

### "Out of memory"
- Close other applications
- Try a shorter video (under 30 seconds)
- Restart the app

### App won't start
- Make sure virtual environment is activated
- You should see `(venv)` in your terminal prompt
- If not, run the activate command again

### Processing takes forever
- Normal! SAM 3 is processing each frame
- Expect ~30 minutes per minute of video on Mac
- M3 Pro/Max will be faster

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

### What to Expect (MacBook 2024)

**M3 Pro/Max:**
- First run: 10 min download
- Processing: ~20-30 min per minute of video
- Quality: Excellent

**M3 Base:**
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

**Issues with access or tokens?**
- Text Sean with a screenshot
- Make sure you completed Step 3 (request access) AND Step 4 (get token)

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

### Checking Your Token

```bash
cat .env
```

Should show: `HUGGINGFACE_TOKEN=hf_...` (your token)

---

## Summary

✅ **Request access** to SAM 3 model (Step 3)
✅ **Get your token** from HuggingFace (Step 4)
✅ **Setup once** (~15 min)
✅ **Create .env file** with YOUR token (Step 6)
✅ **Run app** (30 seconds after first time)
✅ **Process video** (~30 min per min of video)
✅ **Get Adobe-ready masks** instantly

**Questions?** Text Sean!

**Ready to start?** Go back to Step 1! 🚀
