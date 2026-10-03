# optical-079

Local webcam vision for the [voice-079](https://github.com/Lasimeri/voice-079) assistant: live camera windows with a Person-of-Interest style tracking reticle that locks onto a face, a "The Machine" optical-feed look, and a presence gate so the assistant listens only when you are actually there.

Everything runs on your own GPU. Face detection uses OpenCV's OpenCL path (any GPU with an OpenCL driver: NVIDIA, AMD, Intel). No cloud, no uploaded video.

## What it does

- **`facetrack`** - captures a webcam, detects a face on the GPU, draws a white corner-bracket reticle with a crosshair and a `SUBJECT: ADMIN` lock, plus a faint grid, frame, and `079 // OPTICAL FEED` HUD. Streams the annotated video to the player and keeps the newest frame as a photo.
- **`cam079`** - runs one or two webcams in always-on-top windows (inside + outside, side by side), with GPU upscaling (a Jinc scaler plus matched chroma scaling, and the AMD FSR shader) and a low-latency, no-buffer pipeline. The inside camera runs through `facetrack` automatically.
- **`facegate`** - watches the latest frame and writes whether a face is present and facing the screen, so the voice assistant can listen only when you are there (fails open).

## Components

| File | What it is |
| --- | --- |
| `facetrack.cpp` | GPU face tracking + the Machine reticle/HUD; outputs YUV4MPEG2 for mpv |
| `facegate.cpp` | presence/awake gate; writes a one-line state file |
| `cam079` | the camera runner: `split` (two cams), `view` (one), `stop`, styles |
| `cam-style-*.filter` | ffmpeg looks: `admin`, `night`, `machine`, `nexpo`, `color` |
| `shaders/FSR.glsl` | AMD FidelityFX Super Resolution upscaler for mpv (MIT, see below) |

## Requirements

- **OpenCV 5** (or 4) with the `xobjdetect` module and the Haar cascades.
- An **OpenCL** driver for your GPU.
- **ffmpeg**, **mpv**, **v4l-utils**.
- **KDE Plasma / KWin** recommended (window placement uses KWin scripting).

## Build

```bash
cd optical-079
./build.sh
```

That compiles `facetrack` and `facegate`. The build lists the OpenCV libraries by hand on purpose: `pkg-config --libs opencv5` pulls in `opencv_viz`, which needs VTK and fails to link.

> **OpenCV 5 gotcha:** `cv::CascadeClassifier` lives in `<opencv2/xobjdetect.hpp>` and `-lopencv_xobjdetect`, not `objdetect`. Both source files already do this.

## Run

```bash
./cam079 split          # inside + outside cameras, side by side on screen DP-2
./cam079 view           # a single camera
./cam079 stop
CAM079_SCREEN=DP-1 ./cam079 split    # choose the monitor
CAM079_TRACK=0       ./cam079 split  # inside camera without face tracking
```

Run `facetrack` standalone against any camera:

```bash
./facetrack /dev/video0 1280 720 "SUBJECT: ADMIN" | \
  mpv --profile=low-latency --untimed --no-cache --title=cam -
```

## Design notes (hard-won)

- **Each camera is its own capture and window.** Stacking two cameras into one ffmpeg graph makes frames pile up and the two halves drift apart by up to a second. Keep them separate.
- **Never let the player cache a live feed.** mpv's default read-ahead is twenty minutes; with it on, a camera drifts minutes behind. Use `--profile=low-latency --untimed --no-cache` and ffmpeg `-fflags nobuffer -flags low_delay`.
- **GPU does the scaling, CPU does the filtering.** mpv scalers (`ewa_lanczossharp` luma + chroma, Mitchell downscale, sigmoid, dither) run on the GPU; the ffmpeg looks run on the CPU. The chroma scaler matters: cameras record colour at half resolution, and a good chroma scaler smooths it.
- **Detection runs on the GPU** via OpenCV's OpenCL (T-API / `cv::UMat`). The Haar detector needs a roughly frontal, lit face; in profile or darkness it shows `SEARCHING`. For robust locking, swap in the YuNet neural detector (a one-time `.onnx` download).

## Credits

`shaders/FSR.glsl` is AMD's FidelityFX Super Resolution, MIT licensed, Copyright (c) Advanced Micro Devices, Inc. The rest of this repository is under the MIT license (see `LICENSE`).

## Picture enhancement (dark rooms)

`facetrack` lifts dark scenes before it detects faces, so both the picture and the tracking improve. It runs on the GPU through `cv::UMat` when OpenCL is available.

| Setting | Default | What it does |
| --- | --- | --- |
| `FACETRACK_ENHANCE` | on (`0` = raw picture) | turns the whole lift on or off |
| `FACETRACK_CLAHE` | `2.0` | CLAHE clip limit on luminance (local contrast; clipped so sensor noise isn't blown up) |
| `FACETRACK_GAMMA` | `1.3` | gamma lift |
| `FACETRACK_DENOISE` | `0.45` (`1` = off) | temporal noise reduction (running average of frames): removes the grain the lift brings up in a still dark room |

**False-lock guard:** the reticle only locks after 3 detections in a row, and the detector uses `minNeighbors` 6. Without these, brightened noise in an empty dark room was sometimes taken for a face.

## Remote laptops in a 2x2 grid

Show other machines' cameras beside the desktop's own:

1. **On each laptop**, copy `facetrack.cpp` and `laptop/` over and build `facetrack`. Run `laptop/facetrack.sh`, or install `laptop/facetrack.desktop` into `~/.config/autostart` (edit its path) to start at login. Besides its own window, the laptop publishes every second annotated frame to `/dev/shm/facetrack.jpg` (`FACETRACK_STREAM`), written to a temporary name and renamed into place, so a reader never sees half a file.
2. **On the desktop**, `lapcams.sh` pulls each laptop's frames over SSH as a live MJPEG stream into a window ("079 laptop", "079 bedroom"). It reconnects on its own and uses the same GPU scalers and FSR as the desktop cameras. Set laptop addresses with `E16_IP` / `YG6_IP`. The password comes from a private file, as in voice-079 (`LAPTOP_SSH_PASS_FILE`).
3. `cam-grid-place` lays out all four on the camera screen: inside top-left, outside top-right, laptop bottom-left, bedroom bottom-right. 4:3 feeds get a 4:3 window centred in their cell. `cam079` defers to the grid when `$XDG_RUNTIME_DIR/speak-079/camgrid` exists, which `lapcams.sh` creates.

Stop the laptop feeds with `kill -- -$(cat $XDG_RUNTIME_DIR/speak-079/lapcams.pid)`.

## Robustness

- `cam079 split` runs with only one camera. The outside camera joins on its own when plugged in, and its night exposure is re-applied on every (re)start, so a replugged camera doesn't come back on auto exposure.
- Camera device numbers can change when a webcam resets on USB. The runners find cameras by capability each time they (re)start, not by fixed `/dev/videoN` paths.

## Self-arranging layout

`cam-grid-place --watch` loads a resident KWin script that re-lays out the camera windows the moment one appears, closes, retitles or resizes itself, so a camera reconnecting or a laptop coming back online snaps straight into place, with no polling. The layout adapts to how many cameras are present: one fills the screen, two sit side by side, three go two over one (the short row centred), four make a 2x2 grid. Each window keeps its picture's aspect. Run `cam-grid-place` with no argument to lay out once, or `--stop` to unload the script. `lapcams.sh` reconnects a dropped laptop feed by itself: its SSH link feeds mpv through a FIFO, so closing the window ends the link too and the loop starts over.
