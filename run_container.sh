#!/bin/bash
##############################################################################
##                            Run the container                             ##
##############################################################################
SRC_CONTAINER=/home/group1/workspace/src
SRC_HOST="$(pwd)"/src
ASSETS_CONTAINER=/home/group1/workspace/assets
ASSETS_HOST="$(pwd)"/assets
DATA_CONTAINER=/home/group1/workspace/data
DATA_HOST="$(pwd)"/data

docker run \
  --name sorting-via-pushing \
  --privileged \
  --rm \
  -it \
  --net=host \
  -v "$SRC_HOST":"$SRC_CONTAINER":rw \
  -v "$ASSETS_HOST":"$ASSETS_CONTAINER":rw \
  -v "$DATA_HOST":"$DATA_CONTAINER":rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg \
  -e DISPLAY="$DISPLAY" -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
  -e PULSE_SERVER=$PULSE_SERVER \
  -p 6006:6006 \
 rp2024/sorting-via-pushing
