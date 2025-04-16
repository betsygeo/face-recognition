FROM python:3.12-slim

# Install system dependencies (including libGL)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && apt-get clean

# Set workdir
WORKDIR /app

# Copy your code
COPY . /app
COPY .env.local .env.local
COPY firebase-adminsdk.json firebase-adminsdk.json

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run your FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
