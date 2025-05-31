# Use official ROS 2 Humble base with Gazebo support
FROM ros:humble

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ROS_DOMAIN_ID=0 \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Install system dependencies and ROS-Gazebo integration
RUN apt-get update && apt-get install -y --no-install-recommends \
    gazebo \
    ros-humble-gazebo-ros-pkgs \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python libraries
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Set working directory and copy app
WORKDIR /app
COPY . /app

# Create non-root user
RUN useradd -ms /bin/bash appuser && chown -R appuser /app
USER appuser

# Run your Python entry point
CMD ["python3", "main.py"]


