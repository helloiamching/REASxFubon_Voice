# ===== Base image =====
FROM python:3.12-slim

# ===== Environment =====
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# ===== Workdir =====
WORKDIR /app

# ===== Copy dependency file first (for better cache reuse) =====
COPY . .

# ===== System dependencies =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    libpq-dev \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# ===== Python dependencies =====
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ===== Entrypoint =====
EXPOSE 5003

CMD ["python", "app.py"]