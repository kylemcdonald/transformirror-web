# transformirror-web

## Native live installation

This branch includes a single-machine native Transformirror runtime for a 4090-class NVIDIA GPU. It captures a USB webcam with FFmpeg/V4L2, runs SDXL Turbo image-to-image with TAESDXL and stable-fast, displays fullscreen with Pyglet/OpenGL, and exposes realtime control over OSC and HTTP.

The native runtime is intended for live video-filter use:

* webcam input
* fullscreen native display
* default `1280x704` processing, configurable at runtime on a 32-pixel grid up to `1280` per dimension
* OSC control for prompt, seed, strength, blend, passthrough, resolution, steps, and screenshots
* HTTP API and browser control frontend
* no automatic playback
* no automatic prompt cycling
* no audio

### System packages

On Ubuntu, install the base runtime dependencies:

```
sudo apt-get update
sudo apt-get install -y \
  git python3-pip python3-venv python3-dev build-essential cmake ninja-build \
  pkg-config ffmpeg v4l-utils libturbojpeg libgl1 libglib2.0-0 \
  libsm6 libxext6 libxrender1 libjpeg-dev zlib1g-dev libopenblas-dev \
  libx11-dev libxcursor-dev libxi-dev libxrandr-dev libxinerama-dev \
  avahi-daemon libnss-mdns
```

The machine should already have a recent NVIDIA driver installed. This setup has been verified on an RTX 4090 with the 580 driver using CUDA 12.1 PyTorch wheels.

### Python environment

```
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel setuptools
.venv/bin/pip install \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.1.0+cu121' \
  'torchvision==0.16.0+cu121' \
  'xformers==0.0.22.post7' \
  'stable_fast @ https://github.com/chengzeyi/stable-fast/releases/download/v0.0.13.post3/stable_fast-0.0.13.post3+torch210cu121-cp310-cp310-manylinux2014_x86_64.whl'
.venv/bin/pip install -r requirements.txt
```

If the prebuilt stable-fast wheel is not compatible with the local CUDA/PyTorch setup, build stable-fast from source and keep the same PyTorch version unless you are also updating the diffusion code.

### Running manually

Edit `live_config.json` if needed. Important defaults:

```
{
  "width": 1280,
  "height": 704,
  "camera_device": "/dev/video0",
  "camera_backend": "ffmpeg",
  "camera_fps": 30,
  "display_index": 0,
  "fullscreen": true,
  "osc_port": 9000,
  "http_port": 8080
}
```

Start the app:

```
./run-transformirror.sh
```

The first launch downloads `stabilityai/sdxl-turbo` and `madebyollin/taesdxl`. After warmup, a 4090 should process `1280x704` frames in roughly 65-70 ms with the default two-step SDXL Turbo settings.

Diffusion resolution can be changed live. Incoming width and height values are clamped down to multiples of 32 and to a maximum of `1280`. The selected resolution is persisted to `live_config.json` and reused on the next startup. The camera runs at the smallest configured capture mode that can provide a centered crop at least as large as the diffusion resolution; for example, `1280x640` uses a `1280x720` camera mode with a centered `1280x640` crop, and `1024x512` uses a centered `1280x640` crop resized to `1024x512`.

### Systemd service

Install and start the user service:

```
./install-transformirror-service.sh
```

Useful commands:

```
systemctl --user status transformirror.service
journalctl --user -u transformirror.service -f
systemctl --user restart transformirror.service
systemctl --user stop transformirror.service
```

### HTTP control

The HTTP server binds to `0.0.0.0:8080`. Open:

```
http://localhost:8080/
http://<machine-ip>:8080/
http://<hostname>.local:8080/
```

The API state endpoint is:

```
GET /api/state
```

Update controls:

```
curl -X POST http://localhost:8080/api/state \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a neon mirror portrait","seed":42,"strength":0.7,"blend":0.5}'
```

Change resolution:

```
curl -X POST http://localhost:8080/api/resolution \
  -H 'Content-Type: application/json' \
  -d '{"width":1024,"height":512}'
```

Save the currently composed display frame:

```
curl -X POST http://localhost:8080/api/screenshot \
  -H 'Content-Type: application/json' \
  -d '{"path":"/tmp/transformirror.jpg"}'
```

### OSC control

The OSC server binds to `0.0.0.0:9000/udp`, so it is available on LAN IPs, ZeroTier IPs, and `<hostname>.local` when mDNS resolution is available.

Supported addresses:

```
/prompt        string
/seed          int
/strength      float 0..1
/blend         float 0..1    # 0 = raw webcam, 1 = processed output
/passthrough   bool          # true = raw webcam, false = processed output
/resolution    int int       # width height, clamped to 32-pixel steps <= 1280
/resolution    string        # e.g. "1024x512"
/width         int           # updates width, keeping current height
/height        int           # updates height, keeping current width
/steps         int 1..8
/screenshot    string path
```

Namespaced versions also work:

```
/transformirror/prompt
/transformirror/seed
/transformirror/strength
/transformirror/blend
/transformirror/passthrough
/transformirror/resolution
/transformirror/width
/transformirror/height
/transformirror/steps
/transformirror/screenshot
```

### mDNS service discovery

To advertise the control frontend and OSC server with Avahi/Bonjour:

```
sudo install -m 0644 -o root -g root avahi-transformirror.service /etc/avahi/services/transformirror.service
sudo systemctl restart avahi-daemon
```

This publishes:

* `_http._tcp` on port `8080`
* `_osc._udp` on port `9000`

Direct access by IP still works even if mDNS multicast is unavailable on a given network.

Set up NVIDIA drivers:

```
sudo apt update
sudo apt upgrade
sudo apt autoremove
sudo apt autoclean
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-550
sudo reboot now
```

Install CUDA:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.2-560.35.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.2-560.35.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

Edit the ~/.bashrc to add:

```
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Get the code:

```
git clone https://github.com/kylemcdonald/transformirror-web.git
cd transformirror-web
```

Install dependencies:

```
sudo apt install -y python3 python3-pip git libturbojpeg libgl1-mesa-glx libglib2.0-0
pip3 install -r requirements.txt
```

Create self-signed SSL certificates using the IP address from eno1 interface:

```
IP_ADDRESS=$(ip -4 addr show eno1 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/OU=OrganizationalUnit/CN=$IP_ADDRESS"
```

Run the server:

```
python3 server.py
```

The server will run on port 8443. Access the application by opening a web browser and navigating to `https://<your-ip-address>:8443`.

To update the prompt, you can use the `/set` endpoint:

```
http://localhost:8443/set?prompt=your new prompt here
```

Replace "your new prompt here" with the desired prompt text.

## RunPod

```
mkdir /workspace/.cache
export HF_HOME=/workspace/.cache
pip3 install -r requirements.txt
apt update
apt install libturbojpeg
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
