FROM python:3.10-slim

# 1. Install system dependencies (FFmpeg is required for audio)
RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# 2. Setup a non-root user (Required for Hugging Face Spaces permissions)
RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown user:user /app

# 3. Switch to user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/app/cache \
    HF_HOME=/app/cache

# 4. Install Python dependencies
COPY --chown=user:user requirements.txt .

# Install Torch CPU (saves space) - Critical for free tier
RUN pip install --no-cache-dir --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements (Flask, Librosa, Gunicorn)
RUN pip install --no-cache-dir --user -r requirements.txt

# 5. Copy application code
COPY --chown=user:user . .

# 6. Pre-download models (Optional but speeds up first boot)
# We do this AFTER installing requirements so libs are available
RUN python -c "from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration; \
    print('Downloading models...'); \
    AutoModelForAudioClassification.from_pretrained('garystafford/wav2vec2-deepfake-voice-detector'); \
    AutoFeatureExtractor.from_pretrained('garystafford/wav2vec2-deepfake-voice-detector'); \
    WhisperProcessor.from_pretrained('openai/whisper-base'); \
    WhisperForConditionalGeneration.from_pretrained('openai/whisper-base'); \
    print('Models downloaded successfully')"

# 7. Expose the Hugging Face port
EXPOSE 7860

# 8. THE CRITICAL FIX: Use Gunicorn for Flask (Not Uvicorn)
CMD ["python", "-m", "gunicorn", "-b", "0.0.0.0:7860", "app:app", "--workers", "2", "--timeout", "120"]