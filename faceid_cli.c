/* faceid: the people the cameras know (faceid.h).
 *
 *   faceid enroll NAME [-c CAMERA] [-s SECONDS]   learn NAME from a live camera
 *   faceid who [-c CAMERA]                        who that camera sees now
 *   faceid list                                   the people known, their samples
 *   faceid forget NAME                            forget NAME
 *
 * CAMERA is a tracker's name (facetrack's FACETRACK_CAM): inside (the
 * desktop's main camera, the default), glass (the Google Glass's), and so
 * on. A tracker with the SFace model beside it publishes, about twice a
 * second, the embedding of the face it has locked on, how many faces it
 * sees and whether the locked one is the largest; enroll collects those for
 * SECONDS (15) while the locked face is the largest in view (the person at
 * the camera: a poster or a screen behind detects as small faces too),
 * drops the ones that disagree with the rest (a turned head, a
 * blur: cosine to their mean under 0.5), and keeps at least five. The
 * trackers read the gallery again within a second: the new name shows on
 * the reticle at once.
 *
 * Plain C, no OpenCV: build with gcc -O2 -o faceid faceid_cli.c faceid.c -lm.
 */
#define _GNU_SOURCE
#include "faceid.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static faceid_gallery g;

static long long now_us(void) {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (long long) t.tv_sec * 1000000LL + t.tv_nsec / 1000;
}

static int usage(void) {
    fprintf(stderr, "faceid enroll NAME [-c CAMERA] [-s SECONDS]\nfaceid who [-c CAMERA]\nfaceid list\nfaceid forget NAME\n");
    return 1;
}

static int enroll(const char *name, const char *cam, int seconds) {
    static float got[240][FACEID_DIM];
    int n = 0, behind = 0;
    long long last = 0, until = now_us() + (long long) seconds * 1000000;
    printf("faceid: look at the %s camera, nearest to it, for %d s (turn a little, as you would)\n", cam, seconds);
    while (now_us() < until && n < 240) {
        float e[FACEID_DIM];
        long long when;
        int faces, largest;
        if (faceid_read_sample(cam, e, &when, &faces, &largest) == 0 && when != last) {
            last = when;
            if (largest) memcpy(got[n++], e, sizeof e);
            else behind++;
            printf("\r  %d samples%s   ", n, behind ? " (some locked on a face not the nearest: not used)" : "");
            fflush(stdout);
        }
        usleep(100000);
    }
    printf("\n");
    if (n == 0) {
        fprintf(stderr, "faceid: no samples from the %s camera: is its tracker running with the SFace model beside it (%s)?\n",
                cam, last ? "its lock was not on the nearest face" : "nothing published");
        return 1;
    }
    /* Their mean, then only those that agree with it. */
    float mean[FACEID_DIM] = { 0 };
    for (int i = 0; i < n; i++)
        for (int k = 0; k < FACEID_DIM; k++) mean[k] += got[i][k];
    faceid_normalize(mean);
    int kept = 0;
    double self = 0;
    for (int i = 0; i < n; i++) {
        float c = faceid_cosine(got[i], mean);
        if (c < 0.5f) continue;
        self += c;
        if (kept != i) memcpy(got[kept], got[i], sizeof got[0]);
        kept++;
    }
    if (kept < 5) {
        fprintf(stderr, "faceid: only %d of %d samples agree: face the camera, nobody else in view, and try again\n", kept, n);
        return 1;
    }
    float score;
    const char *other = faceid_match(&g, mean, &score);
    if (other && strcmp(other, name))
        printf("faceid: note: this face already matches %s (similarity %.2f)\n", other, score);
    faceid_person *p = faceid_person_get(&g, name);
    if (!p) { fprintf(stderr, "faceid: %s: a name is letters, digits, - and _, and at most %d people\n", name, FACEID_PEOPLE); return 1; }
    /* At most a third of the kept samples for one call, evenly spread: the
     * gallery holds FACEID_SAMPLES, and a later enrolment in other light
     * should not push all of these out. */
    int step = kept > FACEID_SAMPLES / 3 ? kept / (FACEID_SAMPLES / 3) : 1;
    int added = 0;
    for (int i = 0; i < kept; i += step) { faceid_person_add(p, got[i]); added++; }
    if (faceid_save(&g, p) < 0) { fprintf(stderr, "faceid: could not write %s/%s.emb\n", g.dir, name); return 1; }
    printf("faceid: %s: %d samples added (%d now), %d of %d agreed, mean similarity to each other %.2f; saved in %s/%s.emb\n",
           name, added, p->count, kept, n, self / kept, g.dir, name);
    return 0;
}

static int who(const char *cam) {
    float e[FACEID_DIM], score;
    long long when;
    int faces, largest;
    if (faceid_read_sample(cam, e, &when, &faces, &largest) < 0) {
        printf("faceid: nothing from the %s camera (no face locked yet, or no tracker with the model)\n", cam);
        return 1;
    }
    double age = (now_us() - when) / 1e6;
    const char *n = faceid_match(&g, e, &score);
    if (n) printf("%s: %s (similarity %.2f), %d face%s in view, %.1f s ago\n", cam, n, score, faces, faces == 1 ? "" : "s", age);
    else printf("%s: nobody known (best similarity %.2f), %d face%s in view, %.1f s ago\n", cam, score, faces, faces == 1 ? "" : "s", age);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) return usage();
    faceid_load(&g, NULL);
    const char *cmd = argv[1], *name = NULL, *cam = "inside";
    int seconds = 15;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-c") && i + 1 < argc) cam = argv[++i];
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seconds = atoi(argv[++i]);
        else if (!name && argv[i][0] != '-') name = argv[i];
        else return usage();
    }
    if (!strcmp(cmd, "enroll") && name) return enroll(name, cam, seconds < 3 ? 3 : seconds);
    if (!strcmp(cmd, "who")) return who(cam);
    if (!strcmp(cmd, "list")) {
        if (!g.people) printf("faceid: nobody enrolled yet (%s); faceid enroll NAME\n", g.dir);
        for (int i = 0; i < g.people; i++) printf("%-20s %2d samples\n", g.p[i].name, g.p[i].count);
        return 0;
    }
    if (!strcmp(cmd, "forget") && name) {
        if (faceid_forget(&g, name) < 0) { fprintf(stderr, "faceid: %s is not enrolled\n", name); return 1; }
        printf("faceid: %s forgotten\n", name);
        return 0;
    }
    return usage();
}
