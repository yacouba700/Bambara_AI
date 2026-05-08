# ============================================
# VIDEO AI TRANSLATOR - COMPLETE AI BACKEND
# ============================================
# Features:
# - Upload video
# - Extract speech with Whisper
# - Translate text
# - Generate AI voice
# - Merge audio + video
# - Return final dubbed video
#
# Recommended hosting:
# - Render
# - Railway
# - VPS Ubuntu
#
# NOT recommended for:
# - Vercel serverless
#
# ============================================

# INSTALL:
# pip install fastapi uvicorn python-multipart
# pip install openai-whisper edge-tts ffmpeg-python deep-translator

# Linux dependency:
# sudo apt install ffmpeg

# ============================================

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import whisper
import edge_tts
import ffmpeg
import os
import uuid
import asyncio

from deep_translator import GoogleTranslator

# ============================================

app = FastAPI()

# Load Whisper model
model = whisper.load_model("base")

# Folders
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================
# HOME
# ============================================

@app.get("/")
def home():
    return {
        "status": "AI Video Translator Backend Running"
    }

# ============================================
# PROCESS VIDEO
# ============================================

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):

    try:

        # ====================================
        # SAVE VIDEO
        # ====================================

        video_id = str(uuid.uuid4())

        input_video = f"{UPLOAD_FOLDER}/{video_id}.mp4"
        output_audio = f"{OUTPUT_FOLDER}/{video_id}.mp3"
        output_video = f"{OUTPUT_FOLDER}/{video_id}_dubbed.mp4"

        with open(input_video, "wb") as buffer:
            buffer.write(await file.read())

        # ====================================
        # STEP 1 - TRANSCRIPTION
        # ====================================

        result = model.transcribe(input_video)

        original_text = result["text"]

        # ====================================
        # STEP 2 - TRANSLATION
        # ====================================

        # Example:
        # Translate English -> French
        #
        # You can later replace with:
        # - Bambara dataset
        # - M2M100
        # - NLLB Meta AI

        translated_text = GoogleTranslator(
            source='auto',
            target='fr'
        ).translate(original_text)

        # ====================================
        # STEP 3 - AI VOICE
        # ====================================

        communicate = edge_tts.Communicate(
            translated_text,
            voice="fr-FR-DeniseNeural"
        )

        await communicate.save(output_audio)

        # ====================================
        # STEP 4 - MERGE AUDIO + VIDEO
        # ====================================

        video_input = ffmpeg.input(input_video)
        audio_input = ffmpeg.input(output_audio)

        ffmpeg.output(
            video_input.video,
            audio_input.audio,
            output_video,
            vcodec='copy',
            acodec='aac',
            shortest=None
        ).run(overwrite_output=True)

        # ====================================
        # RESPONSE
        # ====================================

        return JSONResponse({
            "success": True,
            "original_text": original_text,
            "translated_text": translated_text,
            "video_url": f"/download/{video_id}"
        })

    except Exception as e:

        return JSONResponse({
            "success": False,
            "error": str(e)
        })

# ============================================
# DOWNLOAD FINAL VIDEO
# ============================================

@app.get("/download/{video_id}")
def download_video(video_id: str):

    video_path = f"{OUTPUT_FOLDER}/{video_id}_dubbed.mp4"

    if os.path.exists(video_path):
        return FileResponse(
            video_path,
            media_type='video/mp4',
            filename="translated_video.mp4"
        )

    return JSONResponse({
        "error": "Video not found"
    })

# ============================================
# RUN SERVER
# ============================================

# START:
# uvicorn main:app --host 0.0.0.0 --port 8000

# ============================================
# PROJECT STRUCTURE
# ============================================

"""
project/

├── main.py
├── uploads/
├── outputs/
├── requirements.txt

"""

# ============================================
# requirements.txt
# ============================================

"""
fastapi
uvicorn
python-multipart
openai-whisper
edge-tts
ffmpeg-python
deep-translator
"""

# ============================================
# DEPLOYMENT (RENDER)
# ============================================

"""
1. Push project to GitHub

2. Go to:
https://render.com

3. New Web Service

4. Connect GitHub repo

5. Build command:
pip install -r requirements.txt

6. Start command:
uvicorn main:app --host 0.0.0.0 --port 10000

"""

# ============================================
# FUTURE UPGRADES
# ============================================

"""
- Bambara AI translation
- Lip sync
- Voice cloning
- Subtitle generation
- TikTok auto upload
- Multiple languages
- AI speaker detection
"""
