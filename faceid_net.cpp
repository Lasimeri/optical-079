// faceid_net.cpp: the one part of faceid not in C, SFace's forward pass,
// because OpenCV's DNN has no C interface. Three C functions (faceid.h):
// open, forward, close.
//
// Where it runs, measured on this desktop (2026-10-07, one 112 x 112 face):
//   CPU (OpenCV 5's new engine)                    7.3 ms
//   RTX 3090 Ti through OpenCL (the classic engine) 97.5 ms
// identical embeddings. OpenCL is no use for a network this small; the GPU
// pays through OpenCV's CUDA backend, which this OpenCV build (extra
// "opencv") lacks and "opencv-cuda" has. So: CUDA when OpenCV has it (a
// device counted), else the CPU. FACEID_DNN=cuda|opencl|cpu forces one.
#include "faceid.h"
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct dnn_box { cv::dnn::Net net; };

// *ON_GPU: 0 the CPU, 1 OpenCL, 2 CUDA.
extern "C" void *faceid_dnn_open(const char *model, int gpu, int *on_gpu, char *err, size_t errlen) {
    try {
        const char *force = std::getenv("FACEID_DNN");
        int mode = 0;
        if (force && !std::strcmp(force, "cuda")) mode = 2;
        else if (force && !std::strcmp(force, "opencl")) mode = 1;
        else if (force && !std::strcmp(force, "cpu")) mode = 0;
        else if (gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) mode = 2;
        if (mode == 1 && !cv::ocl::haveOpenCL()) mode = 0;
        dnn_box *b = new dnn_box;
        // The path as a cv::String: a bare char * picks the in-memory overload.
        b->net = cv::dnn::readNetFromONNX(cv::String(model), mode ? (int) cv::dnn::ENGINE_CLASSIC : (int) cv::dnn::ENGINE_AUTO);
        if (b->net.empty()) { std::snprintf(err, errlen, "%s: not a network OpenCV can read", model); delete b; return nullptr; }
        if (mode == 2) {
            b->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            b->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } else if (mode == 1) {
            cv::ocl::setUseOpenCL(true);
            b->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            b->net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
        }
        *on_gpu = mode;
        return b;
    } catch (const cv::Exception &e) {
        std::snprintf(err, errlen, "%s", e.what());
        return nullptr;
    }
}

extern "C" int faceid_dnn_forward(void *dnn, const float *blob, float *out) {
    try {
        dnn_box *b = static_cast<dnn_box *>(dnn);
        int shape[4] = { 1, 3, 112, 112 };
        cv::Mat in(4, shape, CV_32F, const_cast<float *>(blob));
        b->net.setInput(in);
        cv::Mat o = b->net.forward();
        if ((int) o.total() < FACEID_DIM) return -1;
        cv::Mat f = o.reshape(1, 1);
        if (f.type() != CV_32F) f.convertTo(f, CV_32F);
        std::memcpy(out, f.ptr<float>(), sizeof(float) * FACEID_DIM);
        return 0;
    } catch (const cv::Exception &) {
        return -1;
    }
}

extern "C" void faceid_dnn_close(void *dnn) { delete static_cast<dnn_box *>(dnn); }
