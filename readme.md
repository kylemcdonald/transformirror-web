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

If Python3.10 is already installed:

```
sudo apt install -y python3 python3-pip git libturbojpeg libgl1-mesa-glx libglib2.0-0
pip3 install -r requirements.txt
```

Otherwise, install it with Anaconda:

```
conda create -y -n transformirror-web python=3.10
conda activate transformirror-web
pip install -r requirements.txt
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/OU=Unit/CN=transformirror-web"
python server.py
```