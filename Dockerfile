FROM python:3.11-slim

# Install system libraries needed by OpenCV, MediaPipe, and offline TTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    espeak \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY har_prototype/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code
COPY . .

WORKDIR /app/har_prototype

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=5000

EXPOSE 5000
EXPOSE 8765

CMD ["python", "-u", "main.py"]
