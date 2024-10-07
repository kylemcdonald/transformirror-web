# transformirror-web

Set up NVIDIA drivers and CUDA:

```
sudo apt install nvidia-utils-550-server
```

This code requires Python3.10.

```
conda create -y -n transformirror-web python=3.10
conda activate transformirror-web
git clone https://github.com/kylemcdonald/transformirror-web.git
cd transformirror-web
pip install -r requirements.txt
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/OU=Unit/CN=transformirror-web"
python server.py
```