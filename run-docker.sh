sudo docker run \
    --privileged \
    --rm \
    --gpus all \
    -v "$(pwd)":/home/workspace \
    -p 8443:8443 \
    -p 49152-65535:49152-65535/udp \
    -it \
    pytorch-cuda