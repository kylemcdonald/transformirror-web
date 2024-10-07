sudo docker run \
    --privileged \
    -it \
    --rm \
    --net host \
    --gpus all \
    -v "$(pwd)":/home/workspace \
    transformirror-web