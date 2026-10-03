#!/bin/bash
# Build the optical-079 vision tools: facetrack (GPU face tracking + reticle)
# and facegate (presence gate). OpenCV 4 or 5 with the xobjdetect module.
#
# The OpenCV libraries are listed by hand: `pkg-config --libs opencv5` drags in
# opencv_viz, which needs VTK and fails to link.
set -euo pipefail
cd "$(dirname "$0")"

inc=/usr/include/opencv5
[ -d "$inc" ] || inc=/usr/include/opencv4
libs="-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_xobjdetect"

g++ -O2 -o facetrack facetrack.cpp -I"$inc" $libs
g++ -O2 -o facegate  facegate.cpp  -I"$inc" $libs
echo "built facetrack and facegate"
