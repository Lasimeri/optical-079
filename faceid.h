/* faceid: who a face is. A face's embedding (128 numbers from SFace, the
 * OpenCV zoo's face recognition model, face_recognition_sface_2021dec.onnx)
 * against a gallery of enrolled people: the closest by cosine similarity,
 * or nobody below 0.363 (SFace's own threshold for "the same person").
 *
 * In C, all of it but the network's forward pass (faceid_net.cpp: OpenCV's
 * DNN has no C interface): the alignment of the face to SFace's 112 x 112
 * template from the five landmarks YuNet gives, the network's input, the
 * normalisation, the gallery, the matching, and the live samples a tracker
 * publishes for enrolment (faceid enroll).
 *
 * The gallery: $FACEID_DIR (default ~/.local/share/faceid), one file per
 * person, NAME.emb: "FID1", the dimension (128) and the count as 32-bit
 * integers, then the samples, each FACEID_DIM floats, unit length.
 * The live samples: $XDG_RUNTIME_DIR/faceid/CAM.sample, the latest
 * embedding of the face a tracker has locked on, with when and how many
 * faces it saw.
 */
#ifndef FACEID_H
#define FACEID_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FACEID_DIM 128
#define FACEID_THRESHOLD 0.363f   /* SFace, cosine: at or above, the same person */
#define FACEID_NAME 32
#define FACEID_PEOPLE 64
#define FACEID_SAMPLES 64         /* kept per person, the newest */

typedef struct {
    char name[FACEID_NAME];
    int count;
    float s[FACEID_SAMPLES][FACEID_DIM];
} faceid_person;

typedef struct {
    char dir[512];
    int people;
    faceid_person p[FACEID_PEOPLE];
    long stamp;                    /* the gallery's state when last read (refresh) */
} faceid_gallery;

/* The gallery's directory (FACEID_DIR, else ~/.local/share/faceid). */
const char *faceid_dir(char *buf, size_t len);
/* Reads every NAME.emb in DIR (NULL: faceid_dir). 0 on success, the
 * gallery possibly empty. */
int faceid_load(faceid_gallery *g, const char *dir);
/* Reads it again if a file in it changed (one stat of the directory and
 * its files; call it about once a second). 1 if it was read again. */
int faceid_refresh(faceid_gallery *g);
/* Writes one person's samples (NAME.emb, replaced whole through a
 * temporary file). */
int faceid_save(const faceid_gallery *g, const faceid_person *p);
/* Removes NAME from the gallery and its file. */
int faceid_forget(faceid_gallery *g, const char *name);
/* The person called NAME, made if there is none (NULL when full). */
faceid_person *faceid_person_get(faceid_gallery *g, const char *name);
/* Adds a sample to P, the oldest dropped past FACEID_SAMPLES. */
void faceid_person_add(faceid_person *p, const float e[FACEID_DIM]);

/* Scales E to unit length. */
void faceid_normalize(float e[FACEID_DIM]);
/* Cosine similarity of two unit embeddings. */
float faceid_cosine(const float a[FACEID_DIM], const float b[FACEID_DIM]);
/* The enrolled person closest to E (the best of their samples) and the
 * similarity; NULL when nobody reaches FACEID_THRESHOLD (*score is still
 * the best found, -1 with an empty gallery). */
const char *faceid_match(const faceid_gallery *g, const float e[FACEID_DIM], float *score);

/* Whether NAME is in LIST (names separated by commas or spaces). */
int faceid_allowed(const char *list, const char *name);

/* A tracker's latest sample for CAM: E, when (microseconds), how many
 * faces were in view, and whether the locked face is the largest of them
 * (the person at the camera, not a face on a poster or a screen behind). */
int faceid_publish(const char *cam, const float e[FACEID_DIM], long long when_us, int faces, int largest);
int faceid_read_sample(const char *cam, float e[FACEID_DIM], long long *when_us, int *faces, int *largest);

/* The network (faceid_embed.c with faceid_net.cpp). */
typedef struct faceid_net faceid_net;
/* MODEL: the SFace ONNX file. GPU: 1 to run it on the GPU when OpenCV has
 * CUDA (faceid_net.cpp says why not OpenCL), else the CPU; 0 the CPU;
 * FACEID_DNN=cuda|opencl|cpu forces one. NULL on failure, the reason in
 * ERR. */
faceid_net *faceid_net_open(const char *model, int gpu, char *err, size_t errlen);
void faceid_net_close(faceid_net *n);
/* Where it ended up: 0 the CPU, 1 the GPU through OpenCL, 2 the GPU
 * through CUDA. */
int faceid_net_on_gpu(const faceid_net *n);
const char *faceid_net_where(const faceid_net *n);
/* The embedding of the face in the BGR picture (W x H, STRIDE bytes a
 * row) whose five landmarks LM are (x, y) pairs in YuNet's order: right
 * eye, left eye, nose tip, right and left mouth corners. Unit length.
 * 0 on success. */
int faceid_embed(faceid_net *n, const unsigned char *bgr, int w, int h, int stride,
                 const float lm[10], float out[FACEID_DIM]);

/* The forward pass alone (faceid_net.cpp): BLOB is 1 x 3 x 112 x 112, RGB,
 * 0 to 255; OUT gets FACEID_DIM floats. */
void *faceid_dnn_open(const char *model, int gpu, int *on_gpu, char *err, size_t errlen);
int faceid_dnn_forward(void *dnn, const float *blob, float *out);
void faceid_dnn_close(void *dnn);

#ifdef __cplusplus
}
#endif
#endif
