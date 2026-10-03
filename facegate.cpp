// facegate: the "one degree of separation" between the room and Claude.
// Reads the inside camera's newest frame (a JPG cam079 writes) and detects,
// with OpenCV Haar cascades, whether a face is present and whether its eyes
// are open (facing the camera, awake). It writes a one-line state file the
// voice gate reads, so Claude listens only when the person is actually there
// and facing the screen, not to the room or the TV.
//
// It reads a FILE (not the camera directly), because the camera is held by the
// live view; cam079 writes the newest frame to that file. Mouth-activity needs
// a higher frame rate than the 1 fps file gives, so this version reports
// presence and eyes only; mouth is left for a camera-direct v2.
//
//   facegate [IMAGE] [STATE]
// IMAGE default ~/.cache/voice-079/cam/latest.jpg, STATE default
// $XDG_RUNTIME_DIR/speak-079/facegate. Build: see build.sh (g++, opencv).
#include <opencv2/xobjdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <ctime>
#include <unistd.h>
#include <sys/stat.h>

static std::string env_path(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : fallback;
}

int main(int argc, char** argv) {
    std::string home = env_path("HOME", "");
    std::string img = argc > 1 ? argv[1] : home + "/.cache/voice-079/cam/latest.jpg";
    std::string rt = env_path("XDG_RUNTIME_DIR", "/run/user/1000");
    std::string state = argc > 2 ? argv[2] : rt + "/speak-079/facegate";
    std::string casc = "/usr/share/opencv5/haarcascades/";
    if (access((casc + "haarcascade_frontalface_default.xml").c_str(), R_OK) != 0)
        casc = "/usr/share/opencv4/haarcascades/";

    cv::CascadeClassifier face, eyes;
    if (!face.load(casc + "haarcascade_frontalface_default.xml") ||
        !eyes.load(casc + "haarcascade_eye.xml")) {
        std::fprintf(stderr, "facegate: cannot load cascades from %s\n", casc.c_str());
        return 1;
    }
    std::fprintf(stderr, "facegate: watching %s -> %s\n", img.c_str(), state.c_str());

    time_t last_mtime = 0;
    for (;;) {
        struct stat st;
        if (stat(img.c_str(), &st) != 0) { usleep(300000); continue; }
        if (st.st_mtime == last_mtime) { usleep(200000); continue; }
        last_mtime = st.st_mtime;

        cv::Mat frame = cv::imread(img, cv::IMREAD_GRAYSCALE);
        if (frame.empty()) { usleep(200000); continue; }
        cv::equalizeHist(frame, frame);

        std::vector<cv::Rect> faces;
        face.detectMultiScale(frame, faces, 1.2, 4, 0, cv::Size(80, 80));
        int present = 0, eyes_open = 0, fx = 0, fy = 0, fw = 0, fh = 0;
        if (!faces.empty()) {
            // The largest face.
            cv::Rect f = faces[0];
            for (auto& r : faces) if (r.area() > f.area()) f = r;
            present = 1; fx = f.x; fy = f.y; fw = f.width; fh = f.height;
            // Eyes in the upper half of the face.
            cv::Rect upper(f.x, f.y, f.width, f.height / 2);
            upper &= cv::Rect(0, 0, frame.cols, frame.rows);
            std::vector<cv::Rect> es;
            eyes.detectMultiScale(frame(upper), es, 1.1, 3, 0, cv::Size(f.width / 8, f.width / 8));
            eyes_open = es.size() >= 1 ? 1 : 0;
        }
        FILE* o = std::fopen(state.c_str(), "w");
        if (o) {
            std::fprintf(o, "present=%d eyes=%d face=%d,%d,%d,%d ts=%ld\n",
                         present, eyes_open, fx, fy, fw, fh, (long)time(nullptr));
            std::fclose(o);
        }
    }
    return 0;
}
