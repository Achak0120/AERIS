# ─────────────────────────────────────────────────────────────────────
# ROS 2 (Humble) + Python ML/AI stack for AERIS
# ─────────────────────────────────────────────────────────────────────
FROM ros:humble

# ───── System Environment ────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ROS_DOMAIN_ID=0 \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# ───── Install OS dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-colcon-common-extensions python3-rospy \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 ffmpeg \
    build-essential curl git wget unzip lsb-release \
    && rm -rf /var/lib/apt/lists/*

# ───── Upgrade pip and install Python dependencies ───────────────────
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# ───── Setup workspace ───────────────────────────────────────────────
WORKDIR /app
COPY . /app

# ───── Add non-root user for safety ──────────────────────────────────
RUN useradd -ms /bin/bash appuser && chown -R appuser /app
USER appuser

# ───── Source ROS 2 setup and run Python ─────────────────────────────
SHELL ["/bin/bash", "-c"]
CMD source /opt/ros/humble/setup.bash && python3 main.py

