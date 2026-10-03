#!/bin/bash
# On a laptop: facetrack into a small always-on-top window, and every second
# annotated frame published to $FACETRACK_STREAM for the desktop's lapcams.sh.
# Restarts if the camera or the window goes away.
#   CAM (default /dev/video0)  SIZE (default 1280x720)  TITLE (default laptopcam)
export FACETRACK_STREAM="${FACETRACK_STREAM:-/dev/shm/facetrack.jpg}" FACETRACK_STATUS="${FACETRACK_STATUS:-/dev/shm/facetrack}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}" WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
c=$(dirname "$(readlink -f "$0")")
dev=${CAM:-/dev/video0} size=${SIZE:-1280x720} title=${TITLE:-laptopcam}
while :; do
    "$c/facetrack" "$dev" "${size%x*}" "${size#*x}" "SUBJECT: ADMIN" 2>> "$c/facetrack.log" |
        mpv --really-quiet --no-osc --profile=low-latency --untimed --no-cache \
            --ontop --geometry=360x270+30-70 --title="$title" - 2>> "$c/facetrack.log"
    sleep 1
done
