/* faceid_embed.c: a face's embedding (faceid.h). The face is aligned to
 * SFace's 112 x 112 template the way OpenCV's FaceRecognizerSF::alignCrop
 * does it (a similarity transform, rotation, one scale and a shift, fitted
 * by least squares from YuNet's five landmarks to the template's five
 * points, then a bilinear warp with black outside the picture), turned
 * into the network's input (RGB, planar, 0 to 255, as blobFromImage with
 * swapRB), and run through the network (faceid_net.cpp); the result is
 * scaled to unit length. */
#include "faceid.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct faceid_net {
    void *dnn;
    int gpu;
    float blob[3 * 112 * 112];
};

/* The template: where the eyes, the nose tip and the mouth corners of an
 * aligned face sit in its 112 x 112 crop (OpenCV, face_recognize.cpp). */
static const float tmpl[5][2] = {
    { 38.2946f, 51.6963f }, { 73.5318f, 51.5014f }, { 56.0252f, 71.7366f },
    { 41.5493f, 92.3655f }, { 70.7299f, 92.2041f },
};

faceid_net *faceid_net_open(const char *model, int gpu, char *err, size_t errlen) {
    faceid_net *n = calloc(1, sizeof *n);
    if (!n) { snprintf(err, errlen, "out of memory"); return NULL; }
    n->dnn = faceid_dnn_open(model, gpu, &n->gpu, err, errlen);
    if (!n->dnn) { free(n); return NULL; }
    return n;
}

void faceid_net_close(faceid_net *n) {
    if (!n) return;
    faceid_dnn_close(n->dnn);
    free(n);
}

int faceid_net_on_gpu(const faceid_net *n) { return n->gpu; }

int faceid_embed(faceid_net *n, const unsigned char *bgr, int w, int h, int stride,
                 const float lm[10], float out[FACEID_DIM]) {
    /* The transform: template = [a -b; b a] landmark + t, least squares
     * over the five points about their means. */
    double ms[2] = { 0, 0 }, md[2] = { 0, 0 };
    for (int i = 0; i < 5; i++) {
        ms[0] += lm[2 * i]; ms[1] += lm[2 * i + 1];
        md[0] += tmpl[i][0]; md[1] += tmpl[i][1];
    }
    for (int k = 0; k < 2; k++) { ms[k] /= 5; md[k] /= 5; }
    double A = 0, B = 0, N = 0;
    for (int i = 0; i < 5; i++) {
        double sx = lm[2 * i] - ms[0], sy = lm[2 * i + 1] - ms[1];
        double dx = tmpl[i][0] - md[0], dy = tmpl[i][1] - md[1];
        A += sx * dx + sy * dy;
        B += sx * dy - sy * dx;
        N += sx * sx + sy * sy;
    }
    if (N < 1e-6) return -1;
    double a = A / N, b = B / N, det = a * a + b * b;
    if (det < 1e-12) return -1;
    double tx = md[0] - (a * ms[0] - b * ms[1]), ty = md[1] - (b * ms[0] + a * ms[1]);
    /* Each crop pixel's place in the picture: the inverse transform. */
    double ia = a / det, ib = b / det;
    float *R = n->blob, *G = n->blob + 112 * 112, *Bl = n->blob + 2 * 112 * 112;
    for (int v = 0; v < 112; v++)
        for (int u = 0; u < 112; u++) {
            double qx = u - tx, qy = v - ty;
            double x = ia * qx + ib * qy, y = -ib * qx + ia * qy;
            int x0 = (int) (x < 0 ? x - 1 : x), y0 = (int) (y < 0 ? y - 1 : y);   /* floor */
            double fx = x - x0, fy = y - y0;
            double acc[3] = { 0, 0, 0 };
            for (int j = 0; j < 2; j++)
                for (int i = 0; i < 2; i++) {
                    int px = x0 + i, py = y0 + j;
                    if (px < 0 || py < 0 || px >= w || py >= h) continue;   /* black outside */
                    double wgt = (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
                    const unsigned char *p = bgr + (size_t) py * stride + (size_t) px * 3;
                    acc[0] += wgt * p[0]; acc[1] += wgt * p[1]; acc[2] += wgt * p[2];
                }
            int o = v * 112 + u;
            Bl[o] = (float) acc[0];
            G[o] = (float) acc[1];
            R[o] = (float) acc[2];
        }
    if (faceid_dnn_forward(n->dnn, n->blob, out) < 0) return -1;
    faceid_normalize(out);
    return 0;
}

const char *faceid_net_where(const faceid_net *n) {
    return n->gpu == 2 ? "the GPU (CUDA)" : n->gpu == 1 ? "the GPU (OpenCL)" : "the CPU";
}
