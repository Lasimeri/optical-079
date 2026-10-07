#!/bin/bash
# Build tts079 (the SBTalker voice) and listen079 (whisper.cpp, local).
# WHISPER_DIR: a whisper.cpp checkout built with CUDA (default ~/whisper.cpp).
set -euo pipefail
cd "$(dirname "$0")"
w="${WHISPER_DIR:-$HOME/whisper.cpp}"
[ -f tts079_data.h ] || ./gen-data.sh "${TTS079_ADDON:-$HOME/.cache/voice-079/godot-tts-079/addons/tts_079}" > tts079_data.h
tcc -O2 -o tts079 tts079.c
# Whose voice: sherpa-onnx's C API (speaker embeddings) when it is there
# (~/sherpa-onnx: the release's shared libraries and c-api.h).
s="${SHERPA_ONNX_DIR:-$HOME/sherpa-onnx}"
sl="$s/sherpa-onnx-v1.13.8-linux-x64-shared-no-tts-lib/lib"
spk=()
if [ -f "$s/include/sherpa-onnx/c-api/c-api.h" ] && [ -f "$sl/libsherpa-onnx-c-api.so" ]; then
    spk=(-DHAVE_SPEAKER -I"$s/include" -L"$sl" -lsherpa-onnx-c-api -Wl,-rpath,"$sl")
fi
gcc -O2 -o listen079 listen079.c -I"$w/include" -I"$w/ggml/include" "${spk[@]}" \
    -L"$w/build/bin" -lwhisper -lggml -lggml-base -lm \
    -Wl,-rpath,"$w/build/bin"
# facegate (the camera gate, "one degree of separation"): OpenCV, built when
# the headers are present. Opt-in; it does not change the voice unless
# LISTEN079_FACEGATE=1.
if pkg-config --exists opencv5 2>/dev/null || [ -d /usr/include/opencv5 ]; then
    g++ -O2 -o facegate facegate.cpp -I/usr/include/opencv5 \
        -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_xobjdetect 2>/dev/null &&
        echo "built facegate" || echo "facegate: build skipped"
fi
echo "built tts079 and listen079"
