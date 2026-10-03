// facetrack: the laptop camera with a Person-of-Interest style tracking
// reticle locked on the admin's face. Detection runs on the Radeon 680M
// through OpenCV's OpenCL path (T-API, cv::UMat); the annotated frame is
// written to stdout as a YUV4MPEG2 stream for mpv to show in the laptopcam
// window, and the newest frame is kept as latest.jpg for Claude.
//
//   facetrack [DEVICE] [W] [H] [LABEL]
// DEVICE default /dev/video0, W H default 1280x720, LABEL default
// "SUBJECT: ADMIN". The snapshot path is $FACETRACK_SNAP, default
// ~/.cache/lapcam/latest.jpg. Build: see facetrack.sh.
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/xobjdetect.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
#include <unistd.h>
#include <sys/stat.h>
#include <algorithm>

static std::string cascade_path() {
    const char* c[] = {
        "/usr/share/opencv5/haarcascades/haarcascade_frontalface_alt2.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml",
        "/usr/share/opencv5/haarcascades/haarcascade_frontalface_default.xml", nullptr};
    for (int i = 0; c[i]; i++) if (access(c[i], R_OK) == 0) return c[i];
    return "";
}

// A Machine-style corner bracket at (x,y,w,h): four L corners, a thin frame,
// a centre crosshair. White for the admin (friendly), as the analog interface.
static void reticle(cv::Mat& f, cv::Rect r, cv::Scalar col, int lock) {
    int a = std::max(10, r.width / 5), t = 2;
    cv::Point tl(r.x, r.y), tr(r.x + r.width, r.y), bl(r.x, r.y + r.height), br(r.x + r.width, r.y + r.height);
    // thin full frame, faint
    cv::rectangle(f, r, col * 0.5, 1, cv::LINE_AA);
    // corner Ls
    cv::line(f, tl, tl + cv::Point(a, 0), col, t, cv::LINE_AA); cv::line(f, tl, tl + cv::Point(0, a), col, t, cv::LINE_AA);
    cv::line(f, tr, tr + cv::Point(-a, 0), col, t, cv::LINE_AA); cv::line(f, tr, tr + cv::Point(0, a), col, t, cv::LINE_AA);
    cv::line(f, bl, bl + cv::Point(a, 0), col, t, cv::LINE_AA); cv::line(f, bl, bl + cv::Point(0, -a), col, t, cv::LINE_AA);
    cv::line(f, br, br + cv::Point(-a, 0), col, t, cv::LINE_AA); cv::line(f, br, br + cv::Point(0, -a), col, t, cv::LINE_AA);
    // centre crosshair
    cv::Point c(r.x + r.width / 2, r.y + r.height / 2);
    cv::line(f, c - cv::Point(9, 0), c + cv::Point(9, 0), col, 1, cv::LINE_AA);
    cv::line(f, c - cv::Point(0, 9), c + cv::Point(0, 9), col, 1, cv::LINE_AA);
    // label
    char buf[64];
    std::snprintf(buf, sizeof buf, "ADMIN  %d,%d", c.x, c.y);
    cv::putText(f, buf, cv::Point(r.x, r.y - 8), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(f, buf, cv::Point(r.x, r.y - 8), cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv::LINE_AA);
    (void)lock;
}

int main(int argc, char** argv) {
    // This OpenCV runs its parallel loops on TBB, which ignores
    // OPENCV_FOR_THREADS_NUM: 16 workers at 10-20 % each, 362 % in all
    // (2026-10-03), for loops the GPU does most of the work around. The arena
    // is set here ($FACETRACK_THREADS, default 2).
    cv::setNumThreads(getenv("FACETRACK_THREADS") ? atoi(getenv("FACETRACK_THREADS")) : 2);
    std::string dev = argc > 1 ? argv[1] : "/dev/video0";
    int W = argc > 2 ? atoi(argv[2]) : 1280, H = argc > 3 ? atoi(argv[3]) : 720;
    std::string label = argc > 4 ? argv[4] : "SUBJECT: ADMIN";
    const char* home = getenv("HOME");
    const char* snapenv = getenv("FACETRACK_SNAP");
    std::string snap = snapenv ? std::string(snapenv)
                               : std::string(home ? home : ".") + "/.cache/lapcam/latest.jpg";
    // $FACETRACK_STREAM: every second annotated frame also written there as a
    // JPEG (to a temp name, then renamed, so a reader never sees half a file);
    // another machine can pull it as a live MJPEG feed without opening the
    // camera a second time.
    const char* streamenv = getenv("FACETRACK_STREAM");
    std::string stream = streamenv ? streamenv : "";
    std::string stream_tmp = stream + ".tmp.jpg";
    std::vector<int> jq = {cv::IMWRITE_JPEG_QUALITY, 80};
    // $FACETRACK_STATUS=PREFIX: what this camera saw, in words, for a monitor
    // that cannot look at pictures (the Phi Stream's): once a second PREFIX.status
    // is rewritten (temp name, then renamed) with one line, and the same line
    // is appended to PREFIX.log (moved to PREFIX.log.1 past 2 MB):
    //   t=UNIX_US cam=NAME faces=N locked=0|1 box=X,Y,W,H motion=0..1 light=0..255 fps=N
    // faces: the most found in one frame that second; locked: the reticle on
    // a face; motion: the largest mean frame difference that second (a 96 px
    // wide grey picture, 0 still, 1 every pixel changed fully); light: the
    // mean grey. $FACETRACK_CAM names the camera (default: PREFIX's base name).
    const char* statenv = getenv("FACETRACK_STATUS");
    std::string status = statenv ? statenv : "";
    std::string camname = getenv("FACETRACK_CAM") ? getenv("FACETRACK_CAM")
                        : status.substr(status.find_last_of('/') + 1);
    // Picture enhancement settings: FACETRACK_ENHANCE (default on),
    // FACETRACK_GAMMA (default 1.3, the old "admin" look), FACETRACK_CLAHE
    // (clip limit, default 2.0).
    const char* en = getenv("FACETRACK_ENHANCE");
    bool enhance = !(en && en[0] == '0');
    double gamma = getenv("FACETRACK_GAMMA") ? atof(getenv("FACETRACK_GAMMA")) : 1.3;
    double clip = getenv("FACETRACK_CLAHE") ? atof(getenv("FACETRACK_CLAHE")) : 2.0;
    if (gamma <= 0) gamma = 1.0;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip, cv::Size(8, 8));
    double dn_alpha = getenv("FACETRACK_DENOISE") ? atof(getenv("FACETRACK_DENOISE")) : 1.0;
    if (dn_alpha <= 0 || dn_alpha > 1) dn_alpha = 1.0;
    cv::UMat acc;
    cv::Mat glut(1, 256, CV_8U);
    for (int i = 0; i < 256; i++) glut.at<uchar>(i) = cv::saturate_cast<uchar>(255.0 * std::pow(i / 255.0, 1.0 / gamma));

    cv::ocl::setUseOpenCL(true);
    bool gpu = cv::ocl::useOpenCL();
    std::fprintf(stderr, "facetrack: OpenCL %s\n", gpu ? cv::ocl::Device::getDefault().name().c_str() : "off (CPU)");

    // The detector: YuNet (OpenCV's DNN face detector, a 230 KB ONNX model),
    // on the GPU through OpenCL. The Haar cascade it replaced missed faces
    // turned away or small in a dim room (at the 480 px it ran at, a face
    // across the room was under its 40 px minimum). $FACETRACK_MODEL names
    // the model (default: models/ beside this program); without it, the
    // cascade as before. $FACETRACK_SCORE: the least confidence (0.6).
    std::string exe_dir = ".";
    { char b[4096]; ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
      if (n > 0) { b[n] = 0; exe_dir = std::string(b); exe_dir = exe_dir.substr(0, exe_dir.find_last_of('/')); } }
    std::string model = getenv("FACETRACK_MODEL") ? getenv("FACETRACK_MODEL")
                      : exe_dir + "/models/face_detection_yunet_2023mar.onnx";
    float min_score = getenv("FACETRACK_SCORE") ? (float)atof(getenv("FACETRACK_SCORE")) : 0.6f;
    cv::Ptr<cv::FaceDetectorYN> yunet;
    if (access(model.c_str(), R_OK) == 0) {
        int target = gpu ? cv::dnn::DNN_TARGET_OPENCL : cv::dnn::DNN_TARGET_CPU;
        try {
            yunet = cv::FaceDetectorYN::create(model, "", cv::Size(320, 320), min_score, 0.3f, 50,
                                               cv::dnn::DNN_BACKEND_OPENCV, target);
        } catch (const cv::Exception& e) {
            std::fprintf(stderr, "facetrack: YuNet on %s failed (%s); CPU\n", gpu ? "OpenCL" : "CPU", e.what());
            yunet = cv::FaceDetectorYN::create(model, "", cv::Size(320, 320), min_score, 0.3f, 50);
        }
        std::fprintf(stderr, "facetrack: YuNet %s, score >= %.2f\n", model.c_str(), min_score);
    }
    cv::CascadeClassifier face;
    std::string cp = cascade_path();
    if (!yunet && (cp.empty() || !face.load(cp))) { std::fprintf(stderr, "facetrack: no detector\n"); return 1; }

    cv::VideoCapture cap(dev, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, H);
    // 15 frames a second by default: the tracker shows about 16 anyway, and
    // at 30 the 1080p MJPEG stream, on a USB 2.0 hub shared with the DAC and
    // the mixer, came in cut short (Corrupt JPEG data: premature end, the
    // picture's bottom rows smeared). $FACETRACK_FPS sets it.
    cap.set(cv::CAP_PROP_FPS, getenv("FACETRACK_FPS") ? atoi(getenv("FACETRACK_FPS")) : 15);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    if (!cap.isOpened()) { std::fprintf(stderr, "facetrack: cannot open %s\n", dev.c_str()); return 1; }

    // YUV4MPEG2 header for mpv.
    std::printf("YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420jpeg\n", W, H);
    std::fflush(stdout);

    // detect on a frame scaled to this width, box mapped back (YuNet at 960:
    // a face across the room is still some 50 px)
    // (on the CPU, without OpenCL, 640: the laptops' YuNet at 960 took 376 %
    // of a CPU; never wider than the frame itself, which only adds pixels)
    const int DW = getenv("FACETRACK_DW") ? atoi(getenv("FACETRACK_DW")) : (yunet ? (gpu ? 960 : 640) : 480);
    cv::Mat frame, gray, small, i420;
    cv::UMat grid;   // the overlay grid, on the GPU
    cv::Rect box; bool have = false; int miss = 0, hits = 0; double ema = 0.4;
    time_t last_snap = 0; long fn = 0;
    // the status's second: its largest face count and motion, its frames
    cv::Mat mprev, mcur; int sec_faces = 0, sec_frames = 0; double sec_motion = 0, sec_light = 0;
    time_t last_status = 0;
    const char* dt = "/usr/share/fonts/TTF/DejaVuSansMono.ttf"; (void)dt;

    for (;;) {
        if (!cap.read(frame) || frame.empty()) { if (++miss > 300) break; usleep(10000); continue; }
        if (frame.cols != W || frame.rows != H) { W = frame.cols; H = frame.rows; }

        // Low-light lift, before detection so the tracker sees better too:
        // CLAHE on the luminance (local contrast, clip-limited so sensor noise
        // is not blown up) and a gamma lift, on the GPU through UMat when
        // OpenCL is on. FACETRACK_ENHANCE=0 shows the raw picture.
        // The frame stays on the GPU (OpenCL T-API) from here to the HUD: the
        // lift, the downscale for detection and the grid blend run there; the
        // CPU decodes, draws the text and reticle, and hands the frame to mpv
        // (on the CPU these took 373 % of a 16-thread CPU, 2026-10-03).
        cv::UMat uf; frame.copyTo(uf);
        if (enhance) {
            cv::UMat lab;
            cv::cvtColor(uf, lab, cv::COLOR_BGR2Lab);
            std::vector<cv::UMat> ch; cv::split(lab, ch);
            clahe->apply(ch[0], ch[0]);
            cv::merge(ch, lab);
            cv::cvtColor(lab, uf, cv::COLOR_Lab2BGR);
            cv::LUT(uf, glut, uf);
            // Temporal noise reduction: a running average of frames (EMA),
            // which removes the grain the lift brings up in a still dark room
            // (averaging ~1/alpha frames). Off by default: it smears motion
            // and keeps float frame buffers on the GPU; FACETRACK_DENOISE=0.45
            // turns it on for a still, dark room.
            if (dn_alpha < 1.0) {
                cv::UMat f32; uf.convertTo(f32, CV_32FC3);
                if (acc.empty() || acc.size() != f32.size()) f32.copyTo(acc);
                else cv::accumulateWeighted(f32, acc, dn_alpha);
                acc.convertTo(uf, CV_8UC3);
            }
        }

        double s = (double)std::min(DW, frame.cols) / frame.cols;
        {
            cv::UMat usmall;
            cv::resize(uf, usmall, cv::Size(), s, s, cv::INTER_AREA);
            usmall.copyTo(small);
        }
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        if (yunet) {
            // YuNet: each row x, y, w, h, five landmarks, score; best first
            // by score here, so the face it is surest of is the one tracked.
            cv::Mat det;
            yunet->setInputSize(small.size());
            yunet->detect(small, det);
            std::vector<std::pair<float, cv::Rect>> ranked;
            for (int i = 0; i < det.rows; i++)
                ranked.push_back({det.at<float>(i, 14), cv::Rect((int)det.at<float>(i, 0), (int)det.at<float>(i, 1),
                                                                (int)det.at<float>(i, 2), (int)det.at<float>(i, 3))});
            std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.first > b.first; });
            for (auto& r : ranked) faces.push_back(r.second);
        } else {
            cv::UMat ug; gray.copyTo(ug);           // UMat -> detect on the GPU via OpenCL
            cv::equalizeHist(ug, ug);
            // minNeighbors 6 (was 4): the brightened dark rooms gave noise that a
            // looser detector called a face.
            face.detectMultiScale(ug, faces, 1.2, 6, 0, cv::Size(40, 40));
            std::sort(faces.begin(), faces.end(), [](auto& a, auto& b) { return a.area() > b.area(); });
        }

        const int nfaces = (int)faces.size();
        // Locked, it stays on its face: the detection nearest the box (its
        // centre within the box's width of the box's centre), not the best
        // scored, which jumped between two faces each frame; past that none
        // counts as a miss, and after the misses the best is taken again.
        if (have && !faces.empty()) {
            cv::Point2d bc(box.x + box.width / 2.0, box.y + box.height / 2.0);
            int pick = -1; double best = 1e18;
            for (size_t i = 0; i < faces.size(); i++) {
                const cv::Rect& f = faces[i];
                cv::Point2d fc((f.x + f.width / 2.0) / s, (f.y + f.height / 2.0) / s);
                double d = std::hypot(fc.x - bc.x, fc.y - bc.y);
                if (d < box.width && d < best) { best = d; pick = (int)i; }
            }
            if (pick < 0) faces.clear(); else std::swap(faces[0], faces[pick]);
        }
        if (!faces.empty()) {
            cv::Rect f = faces[0];
            cv::Rect det((int)(f.x / s), (int)(f.y / s), (int)(f.width / s), (int)(f.height / s));
            // Lock only after detections in a row (3 for the cascade, 2 for
            // YuNet, which seldom fires on noise): a one-frame false hit in
            // the noise never draws the reticle.
            if (!have) { if (++hits >= (yunet ? 2 : 3)) { box = det; have = true; } }
            else {   // smooth to kill jitter
                box.x = (int)(ema * det.x + (1 - ema) * box.x);
                box.y = (int)(ema * det.y + (1 - ema) * box.y);
                box.width = (int)(ema * det.width + (1 - ema) * box.width);
                box.height = (int)(ema * det.height + (1 - ema) * box.height);
            }
            miss = 0;
        } else { hits = 0; if (have && ++miss > 20) have = false; }

        if (!status.empty()) {
            cv::resize(gray, mcur, cv::Size(96, std::max(1, gray.rows * 96 / std::max(1, gray.cols))), 0, 0, cv::INTER_AREA);
            if (!mprev.empty() && mprev.size() == mcur.size()) {
                cv::Mat d; cv::absdiff(mcur, mprev, d);
                sec_motion = std::max(sec_motion, cv::mean(d)[0] / 255.0);
            }
            mcur.copyTo(mprev);
            sec_light = cv::mean(mcur)[0];
            sec_faces = std::max(sec_faces, nfaces);
            sec_frames++;
            time_t tnow = time(nullptr);
            if (tnow != last_status) {
                if (last_status != 0) {
                    struct timespec tsp; clock_gettime(CLOCK_REALTIME, &tsp);
                    char line[256];
                    cv::Rect b = have ? (box & cv::Rect(0, 0, W, H)) : cv::Rect();
                    std::snprintf(line, sizeof line,
                        "t=%lld cam=%s faces=%d locked=%d box=%d,%d,%d,%d motion=%.3f light=%d fps=%d\n",
                        (long long)tsp.tv_sec * 1000000LL + tsp.tv_nsec / 1000, camname.c_str(), sec_faces,
                        have ? 1 : 0, b.x, b.y, b.width, b.height, sec_motion, (int)sec_light, sec_frames);
                    std::string tmp = status + ".status.tmp";
                    if (FILE* f = std::fopen(tmp.c_str(), "w")) {
                        std::fputs(line, f); std::fclose(f);
                        std::rename(tmp.c_str(), (status + ".status").c_str());
                    }
                    std::string lg = status + ".log";
                    struct stat stl;
                    if (stat(lg.c_str(), &stl) == 0 && stl.st_size > 2 * 1024 * 1024)
                        std::rename(lg.c_str(), (lg + ".1").c_str());
                    if (FILE* f = std::fopen(lg.c_str(), "a")) { std::fputs(line, f); std::fclose(f); }
                }
                last_status = tnow; sec_faces = 0; sec_frames = 0; sec_motion = 0;
            }
        }

        // The Machine's optical overlay: faint grid, a thin frame.
        // The grid is drawn once and added faintly on the GPU (it was a full
        // copy of the frame and a CPU blend every frame); the frame comes
        // down once for the text and the reticle.
        {
            if (grid.empty() || grid.size() != uf.size()) {
                cv::Mat g(uf.size(), CV_8UC3, cv::Scalar::all(0));
                for (int x = 80; x < W; x += 80) cv::line(g, cv::Point(x, 0), cv::Point(x, H), cv::Scalar(255, 255, 255), 1);
                for (int y = 80; y < H; y += 80) cv::line(g, cv::Point(0, y), cv::Point(W, y), cv::Scalar(255, 255, 255), 1);
                g.copyTo(grid);
            }
            cv::addWeighted(uf, 1.0, grid, 0.06, 0, uf);
            uf.copyTo(frame);
            cv::rectangle(frame, cv::Rect(1, 1, W - 2, H - 2), cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
        // HUD: timestamp top, label bottom (yellow), tracking state
        char ts[32]; time_t now = time(nullptr); struct tm tmv; localtime_r(&now, &tmv);
        std::strftime(ts, sizeof ts, "%H:%M:%S", &tmv);
        char top[80]; std::snprintf(top, sizeof top, "888 // OPTICAL FEED // %s", ts);
        cv::putText(frame, top, cv::Point(24, 34), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(frame, top, cv::Point(24, 34), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::putText(frame, label, cv::Point(24, H - 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(frame, label, cv::Point(24, H - 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 208, 255), 1, cv::LINE_AA);
        const char* st = have ? "TRACKING" : "SEARCHING";
        int bl = 0; cv::Size sz = cv::getTextSize(st, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &bl);
        cv::putText(frame, st, cv::Point(W - sz.width - 24, H - 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(frame, st, cv::Point(W - sz.width - 24, H - 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, have ? cv::Scalar(255, 255, 255) : cv::Scalar(120, 120, 120), 1, cv::LINE_AA);
        if (have) reticle(frame, box & cv::Rect(0, 0, W, H), cv::Scalar(255, 255, 255), 1);

        // newest frame for Claude, ~1/s
        if (now != last_snap) { last_snap = now; cv::imwrite(snap, frame); }
        // the live feed for another machine
        if (!stream.empty() && (fn % 2) == 0 && cv::imwrite(stream_tmp, frame, jq))
            std::rename(stream_tmp.c_str(), stream.c_str());

        // out as I420 y4m
        cv::cvtColor(frame, i420, cv::COLOR_BGR2YUV_I420);
        std::fputs("FRAME\n", stdout);
        std::fwrite(i420.data, 1, (size_t)W * H * 3 / 2, stdout);
        if (ferror(stdout)) break;   // mpv gone
        fn++;
    }
    std::fprintf(stderr, "facetrack: ended after %ld frames\n", fn);
    return 0;
}
