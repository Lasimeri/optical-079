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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
#include <unistd.h>

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

    cv::CascadeClassifier face;
    std::string cp = cascade_path();
    if (cp.empty() || !face.load(cp)) { std::fprintf(stderr, "facetrack: no cascade\n"); return 1; }

    cv::VideoCapture cap(dev, cv::CAP_V4L2);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, H);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    if (!cap.isOpened()) { std::fprintf(stderr, "facetrack: cannot open %s\n", dev.c_str()); return 1; }

    // YUV4MPEG2 header for mpv.
    std::printf("YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420jpeg\n", W, H);
    std::fflush(stdout);

    // detect on a frame scaled to this width, box mapped back
    const int DW = 480;
    cv::Mat frame, gray, small, i420;
    cv::Rect box; bool have = false; int miss = 0, hits = 0; double ema = 0.4;
    time_t last_snap = 0; long fn = 0;
    const char* dt = "/usr/share/fonts/TTF/DejaVuSansMono.ttf"; (void)dt;

    for (;;) {
        if (!cap.read(frame) || frame.empty()) { if (++miss > 300) break; usleep(10000); continue; }
        if (frame.cols != W || frame.rows != H) { W = frame.cols; H = frame.rows; }

        // Low-light lift, before detection so the tracker sees better too:
        // CLAHE on the luminance (local contrast, clip-limited so sensor noise
        // is not blown up) and a gamma lift, on the GPU through UMat when
        // OpenCL is on. FACETRACK_ENHANCE=0 shows the raw picture.
        if (enhance) {
            cv::UMat uf, lab; frame.copyTo(uf);
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
            uf.copyTo(frame);
        }

        double s = (double)DW / frame.cols;
        cv::resize(frame, small, cv::Size(), s, s, cv::INTER_AREA);
        cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
        cv::UMat ug; gray.copyTo(ug);           // UMat -> detect on the GPU via OpenCL
        cv::equalizeHist(ug, ug);
        std::vector<cv::Rect> faces;
        // minNeighbors 6 (was 4): the brightened dark rooms gave noise that a
        // looser detector called a face.
        face.detectMultiScale(ug, faces, 1.2, 6, 0, cv::Size(40, 40));

        if (!faces.empty()) {
            cv::Rect f = faces[0];
            for (auto& r : faces) if (r.area() > f.area()) f = r;
            cv::Rect det((int)(f.x / s), (int)(f.y / s), (int)(f.width / s), (int)(f.height / s));
            // Lock only after 3 detections in a row: a one-frame false hit
            // in the noise never draws the reticle.
            if (!have) { if (++hits >= 3) { box = det; have = true; } }
            else {   // smooth to kill jitter
                box.x = (int)(ema * det.x + (1 - ema) * box.x);
                box.y = (int)(ema * det.y + (1 - ema) * box.y);
                box.width = (int)(ema * det.width + (1 - ema) * box.width);
                box.height = (int)(ema * det.height + (1 - ema) * box.height);
            }
            miss = 0;
        } else { hits = 0; if (have && ++miss > 20) have = false; }

        // The Machine's optical overlay: faint grid, a thin frame.
        {   // grid + frame drawn at low alpha via a blended copy
            cv::Mat ov = frame.clone();
            for (int x = 80; x < W; x += 80) cv::line(ov, cv::Point(x, 0), cv::Point(x, H), cv::Scalar(255, 255, 255), 1);
            for (int y = 80; y < H; y += 80) cv::line(ov, cv::Point(0, y), cv::Point(W, y), cv::Scalar(255, 255, 255), 1);
            cv::addWeighted(ov, 0.06, frame, 0.94, 0, frame);
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
