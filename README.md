# optical-079

Local webcam vision for the [voice-079](https://github.com/Lasimeri/voice-079) assistant: live camera windows with a Person-of-Interest style tracking reticle that locks onto a face, a "The Machine" optical-feed look, and a presence gate so the assistant listens only when you are actually there.

Everything runs on your own GPU. Face detection uses OpenCV's OpenCL path (any GPU with an OpenCL driver: NVIDIA, AMD, Intel). No cloud, no uploaded video.

## What it does

- **`facetrack`** - captures a webcam, detects a face on the GPU, draws a white corner-bracket reticle with a crosshair and a `SUBJECT: ADMIN` lock, plus a faint grid, frame, and `888 // OPTICAL FEED` HUD. Streams the annotated video to the player and keeps the newest frame as a photo.
- **`cam079`** - runs one or two webcams in always-on-top windows (inside + outside, side by side), with GPU upscaling (a Jinc scaler plus matched chroma scaling, and the AMD FSR shader) and a low-latency, no-buffer pipeline. The inside camera runs through `facetrack` automatically.
- **`facegate`** - watches the latest frame and writes whether a face is present and facing the screen, so the voice assistant can listen only when you are there (fails open).

## Components

| File | What it is |
| --- | --- |
| `facetrack.cpp` | GPU face tracking + the Machine reticle/HUD; outputs YUV4MPEG2 for mpv |
| `facegate.cpp` | presence/awake gate; writes a one-line state file |
| `faceid.c`, `faceid.h`, `faceid_embed.c`, `faceid_net.cpp` | who a face is: a gallery of the people you enrol (one file each in `~/.local/share/faceid`, never in this repository), cosine matching of SFace embeddings, YuNet's five landmarks aligned to SFace's template in C; only the network's forward pass is C++ (OpenCV DNN, CUDA when OpenCV has it) |
| `faceid_cli.c` | `faceid enroll NAME`, `who`, `list`, `forget`: enrolment from a running tracker, plain C |
| `build-face.sh` | builds `faceid` and `facetrack` with the face identity (the models in `models/`: YuNet and SFace from the OpenCV model zoo, Apache 2.0) |
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
| `FACETRACK_DENOISE` | off (`0.45` = on) | temporal noise reduction (running average of frames). Off by default: it smears motion and keeps float frame buffers on the GPU. Turn it on only for a still, dark room |

**False-lock guard:** the reticle only locks after 3 detections in a row, and the detector uses `minNeighbors` 6. Without these, brightened noise in an empty dark room was sometimes taken for a face.

## Remote laptops in a 2x2 grid

Show other machines' cameras beside the desktop's own:

1. **On each laptop**, copy `facetrack.cpp` and `laptop/` over and build `facetrack`. Run `laptop/facetrack.sh`, or install `laptop/facetrack.desktop` into `~/.config/autostart` (edit its path) to start at login. Besides its own window, the laptop publishes every second annotated frame to `/dev/shm/facetrack.jpg` (`FACETRACK_STREAM`), written to a temporary name and renamed into place, so a reader never sees half a file.
2. **On the desktop**, `lapcams.sh` pulls each laptop's frames over SSH as a live MJPEG stream into a window ("079 laptop", "079 bedroom"). It reconnects on its own and uses the same GPU scalers and FSR as the desktop cameras. The laptops' addresses and login live in a local file outside the repository, `~/.config/voice-079/lapcams.conf` (`E16_IP=...`, `YG6_IP=...`, optionally `LAPTOP_USER`, `LAPTOP_SSH_PASS_FILE`; `LAPCAMS_CONF` names another); the password comes from a private file, as in voice-079.
3. `cam-grid-place` lays out all four on the camera screen: inside top-left, outside top-right, laptop bottom-left, bedroom bottom-right. 4:3 feeds get a 4:3 window centred in their cell. `cam079` defers to the grid when `$XDG_RUNTIME_DIR/speak-079/camgrid` exists, which `lapcams.sh` creates.

Stop the laptop feeds with `kill -- -$(cat $XDG_RUNTIME_DIR/speak-079/lapcams.pid)`.

## Robustness

- `cam079 split` runs with only one camera. The outside camera joins on its own when plugged in, and its night exposure is re-applied on every (re)start, so a replugged camera doesn't come back on auto exposure.
- Camera device numbers can change when a webcam resets on USB. The runners find cameras by capability each time they (re)start, not by fixed `/dev/videoN` paths.

## Self-arranging layout

`cam-grid-place --watch` loads a resident KWin script that re-lays out the camera windows the moment one appears, closes, retitles or resizes itself, so a camera reconnecting or a laptop coming back online snaps straight into place, with no polling. The layout adapts to how many cameras are present: one fills the screen, two sit side by side, three go two over one (the short row centred), four make a 2x2 grid. Each window keeps its picture's aspect. Run `cam-grid-place` with no argument to lay out once, or `--stop` to unload the script. `lapcams.sh` reconnects a dropped laptop feed by itself: its SSH link feeds mpv through a FIFO, so closing the window ends the link too and the loop starts over.

**Layouts.** `CAM079_LAYOUT=featured` (the default) puts the first camera present (the inside one, the person) large on the left two-thirds and stacks the rest down the right. Two cameras sit side by side, and one fills the screen. `CAM079_LAYOUT=grid` gives the even grid instead (2 side by side, 3 two over one, 4 a 2x2 grid).

## Kept across reboots

`state079` (shared with voice-079) keeps, on disk, what was last chosen: the cameras (`cam079 split|view|start|stop`), the laptop feeds (`lapcams.sh` and `lapcams.sh stop`), the layout and the screen (`cam-grid-place --layout featured|grid`, `--screen DP-2`, kept and applied at once). The login script brings back exactly that. Only those commands write the state, never a stop at shutdown. The resident layout script also lays out again when the monitors change (`screensChanged`), so a camera screen that comes up after the script is loaded still gets its cameras.

## Detection, status lines, clean frames (2026-10-03)

- **YuNet on the GPU.** `facetrack` detects with OpenCV's YuNet DNN face detector (`models/face_detection_yunet_2023mar.onnx`, 230 KB, OpenCL on the RTX 3090 Ti; CPU on a laptop without OpenCL) at 960 px wide, score at least 0.6 (`FACETRACK_SCORE`). The Haar cascade it replaced missed faces turned away or across a dim room (at 480 px a far face was under its 40 px minimum); it remains the fallback without the model. Locked, the reticle stays on the detection nearest its box instead of the best-scored one, which jumped between faces. Link with `-lopencv_objdetect -lopencv_dnn -lopencv_flann -lopencv_features -lopencv_calib -lopencv_geometry` besides the earlier libraries.
- **15 fps capture** (`FACETRACK_FPS`). At 30 the 1080p MJPEG stream, on a USB 2.0 hub shared with a DAC and a mixer, arrived cut short ("Corrupt JPEG data: premature end", the bottom rows smeared): 17 corrupt frames in 30 s at 30 fps, none at 15. The tracker shows about 16 a second either way.
- **Status lines for a text monitor** (`FACETRACK_STATUS=PREFIX`): once a second `PREFIX.status` is rewritten and `PREFIX.log` appended with `t=UNIX_US cam=NAME faces=N locked=0|1 box=X,Y,W,H motion=0..1 light=0..255 fps=N`. `cam079` writes the inside camera's into the Phi Stream's workspace (`CAM079_FEEDS`, default `~/.local/share/phi-stream/dev/feeds`); the laptops write `/dev/shm/facetrack.status`, and `lapcams.sh` copies each into the same feed as `laptop` and `bedroom`.
