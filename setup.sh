# wget https://raw.githubusercontent.com/kylemcdonald/transformirror-web/refs/heads/main/setup.sh | bash

git clone https://github.com/kylemcdonald/transformirror-web.git
cd transformirror-web
pip3 install -r requirements.txt

IP_ADDRESS=$(ip -4 addr show eno1 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/OU=OrganizationalUnit/CN=$IP_ADDRESS"
python3 download_files.py

bash run-workers.sh &
python3 server.py &