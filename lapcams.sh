#!/bin/bash
# lapcams.sh: both laptops' face-tracked cameras on this desktop, beside its
# own two, in a 2x2 grid (cam-grid-place). Each laptop's facetrack writes every
# second annotated frame to /dev/shm/facetrack.jpg ($FACETRACK_STREAM, renamed
# into place so a reader never sees half a file); this pulls that file over
# SSH as a live MJPEG stream into an mpv window (low latency, no cache, the
# same GPU scalers and FSR as the desktop cameras). Reconnects on its own.
#   start: setsid lapcams.sh &      stop: lapcams.sh stop
# Either is kept for the next login (state079): a stop stays stopped.
set -u
here=$(dirname "$(readlink -f "$0")")
rt="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/speak-079"
mkdir -p "$rt"
st="$here/state079"
export CAM079_SCREEN="${CAM079_SCREEN:-$("$st" get screen DP-2)}"
if [ "${1:-}" = stop ]; then
    [ -f "$rt/lapcams.pid" ] && kill -- -"$(cat "$rt/lapcams.pid")" 2>/dev/null
    rm -f "$rt/lapcams.pid"
    "$st" set lapcams off
    echo "lapcams: off"
    exit 0
fi
"$st" set lapcams on
echo $$ > "$rt/lapcams.pid"
touch "$rt/camgrid"
# The laptops' SSH password stays out of the code: a private file
# ($LAPTOP_SSH_PASS_FILE, default ~/.config/voice-079/laptop-ssh-pass, mode
# 600) that a tiny askpass helper reads. SSH keys would be better still.
pf="${LAPTOP_SSH_PASS_FILE:-$HOME/.config/voice-079/laptop-ssh-pass}"
A="$HOME/.cache/voice-079/laptop-askpass.sh"
umask 077; printf '#!/bin/sh\nexec cat "%s"\n' "$pf" > "$A"; chmod 700 "$A"
MPV_HQ="--scale=ewa_lanczossharp --cscale=ewa_lanczossharp --dscale=mitchell \
 --correct-downscaling=yes --linear-downscaling=yes \
 --sigmoid-upscaling=yes --dither-depth=auto --temporal-dither=yes"
pull() {   # IP TITLE
    # ssh feeds mpv through a FIFO, not a pipe: if the window is closed, ssh
    # does not notice for a long time (its writes just fail), and a pipeline
    # would wait on it, so the feed never came back. Here, whichever end goes
    # first, the other is ended and the loop reconnects.
    local fifo="$rt/lapcam-${2// /_}.fifo" sp
    while :; do
        rm -f "$fifo"; mkfifo "$fifo"
        env SSH_ASKPASS="$A" SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
            ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no \
                -o NumberOfPasswordPrompts=1 -o ConnectTimeout=10 \
                -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o StrictHostKeyChecking=no "lasimeri@$1" \
            "bash -c 'while :; do cat /dev/shm/facetrack.jpg 2>/dev/null; sleep 0.066; done'" > "$fifo" 2>/dev/null &
        sp=$!
        mpv --really-quiet --profile=low-latency --untimed --no-cache \
            --demuxer-lavf-format=mjpeg $MPV_HQ --glsl-shaders="$here/shaders/FSR.glsl" \
            --ontop --no-border --screen-name="${CAM079_SCREEN:-DP-2}" \
            --title="$2" - < "$fifo" 2>/dev/null
        kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
        sleep 2
    done
}
pull "${E16_IP:-192.168.0.78}" "079 laptop" &
pull "${YG6_IP:-192.168.0.125}" "079 bedroom" &
( for t in 2 2 3 5 8; do sleep "$t"; "$here/cam-grid-place"; done ) &
wait
