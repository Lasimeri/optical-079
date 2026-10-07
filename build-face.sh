#!/bin/bash
# build-face.sh: the face tools.
#   faceid      who a face is: the gallery, enrolment, matching (C, no OpenCV)
#   facetrack   the cameras' face tracker (C++ for OpenCV's capture, YuNet
#               and drawing; who a face is through faceid's C library)
# Each binary is built beside itself and moved into place, so a running
# tracker keeps the one it started with. The models live in models/:
# face_detection_yunet_2023mar.onnx and face_recognition_sface_2021dec.onnx
# (the OpenCV zoo; SFace sha256 0ba9fbfa...4e79, as its Git LFS pointer).
# Not pkg-config: opencv5's pulls in opencv_viz/VTK and fails to link.
set -euo pipefail
cd "$(dirname "$0")"
cv=/usr/include/opencv5
gcc -O2 -Wall -c faceid.c -o faceid.o
gcc -O2 -Wall -c faceid_embed.c -o faceid_embed.o
g++ -O2 -Wall -c faceid_net.cpp -I"$cv" -o faceid_net.o
gcc -O2 -Wall -o faceid.new faceid_cli.c faceid.o -lm
mv faceid.new faceid
g++ -O2 -o facetrack.new facetrack.cpp faceid.o faceid_embed.o faceid_net.o -I"$cv" \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_xobjdetect \
    -lopencv_objdetect -lopencv_dnn -lopencv_flann -lopencv_features -lopencv_calib -lopencv_geometry
mv facetrack.new facetrack
echo "built faceid and facetrack"
