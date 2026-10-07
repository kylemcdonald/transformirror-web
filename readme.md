# transformirror-web

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

Create a pod from `runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04` with HTTP port `8443` exposed, then:

```
cd /workspace
git clone https://github.com/kylemcdonald/transformirror-web.git
cd transformirror-web
./setup-runpod.sh
```

This installs the pinned packages from `requirements-runpod.txt` with uv while the models download in parallel (about a minute on a fast pod), then starts one worker per GPU plus the server in tmux sessions (logs in `logs/`) and waits for every GPU to finish warming up. RunPod's proxy provides HTTPS, so open `https://<pod-id>-8443.proxy.runpod.net/` (or `/buffered` for the jitter-buffered frontend). `/health` reports the number of connected clients and seconds since the last frame.

Pods launched by [transformirror-control](https://github.com/kylemcdonald/transformirror-control) run `runpod-bootstrap.sh` at container start, which clones this repo, runs `setup-runpod.sh`, and reports each stage back to the control app.
