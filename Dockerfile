# Use NVIDIA's official CUDA base image with Python 3.10
FROM nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04

# # Install Python and necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    libturbojpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /home/app
WORKDIR /home/app
RUN pip3 install -r requirements.txt

# Generate SSL certificates
RUN openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/OU=Unit/CN=webrtc-bot"

# Expose web app port
EXPOSE 8443

# Expose WebRTC ports (UDP)
EXPOSE 49152-65535/udp

# Run server.py when the container launches
CMD ["python", "server.py"]