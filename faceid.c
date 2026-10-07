/* faceid.c: the gallery, the matching and the live samples (faceid.h). */
#define _GNU_SOURCE
#include "faceid.h"
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

const char *faceid_dir(char *buf, size_t len) {
    const char *d = getenv("FACEID_DIR");
    if (d && *d) snprintf(buf, len, "%s", d);
    else snprintf(buf, len, "%s/.local/share/faceid", getenv("HOME") ? getenv("HOME") : ".");
    return buf;
}

void faceid_normalize(float e[FACEID_DIM]) {
    double s = 0;
    for (int i = 0; i < FACEID_DIM; i++) s += (double) e[i] * e[i];
    s = sqrt(s);
    if (s > 0) for (int i = 0; i < FACEID_DIM; i++) e[i] = (float) (e[i] / s);
}

float faceid_cosine(const float a[FACEID_DIM], const float b[FACEID_DIM]) {
    double s = 0;
    for (int i = 0; i < FACEID_DIM; i++) s += (double) a[i] * b[i];
    return (float) s;
}

/* A name is letters, digits, '-' and '_': it becomes a file name. */
static int name_ok(const char *n) {
    if (!*n || strlen(n) >= FACEID_NAME) return 0;
    for (const char *c = n; *c; c++)
        if (!((*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') || (*c >= '0' && *c <= '9') || *c == '-' || *c == '_')) return 0;
    return 1;
}

/* The directory's state: the newest change time among it and its files,
 * and their number, folded together. */
static long dir_stamp(const char *dir) {
    struct stat st;
    if (stat(dir, &st) < 0) return 0;
    long s = (long) st.st_mtim.tv_sec * 1000 + st.st_mtim.tv_nsec / 1000000;
    DIR *d = opendir(dir);
    if (!d) return s;
    struct dirent *e;
    char p[1024];
    while ((e = readdir(d))) {
        size_t l = strlen(e->d_name);
        if (l < 5 || strcmp(e->d_name + l - 4, ".emb")) continue;
        snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
        if (stat(p, &st) == 0) s = s * 31 + (long) st.st_mtim.tv_sec * 1000 + st.st_mtim.tv_nsec / 1000000 + (long) st.st_size;
    }
    closedir(d);
    return s;
}

static int read_person(const char *path, faceid_person *p) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    uint32_t dim = 0, count = 0;
    int ok = fread(magic, 1, 4, f) == 4 && !memcmp(magic, "FID1", 4) &&
             fread(&dim, 4, 1, f) == 1 && fread(&count, 4, 1, f) == 1 && dim == FACEID_DIM;
    if (ok) {
        if (count > FACEID_SAMPLES) {   /* keep the newest */
            fseek(f, (long) (count - FACEID_SAMPLES) * FACEID_DIM * 4, SEEK_CUR);
            count = FACEID_SAMPLES;
        }
        ok = fread(p->s, sizeof(float) * FACEID_DIM, count, f) == count;
        p->count = (int) count;
    }
    fclose(f);
    if (!ok) return -1;
    for (int i = 0; i < p->count; i++) faceid_normalize(p->s[i]);
    return 0;
}

int faceid_load(faceid_gallery *g, const char *dir) {
    char buf[512];
    if (!dir) dir = faceid_dir(buf, sizeof buf);
    snprintf(g->dir, sizeof g->dir, "%s", dir);
    g->people = 0;
    g->stamp = dir_stamp(g->dir);
    DIR *d = opendir(g->dir);
    if (!d) return errno == ENOENT ? 0 : -1;
    struct dirent *e;
    char p[1024];
    while ((e = readdir(d)) && g->people < FACEID_PEOPLE) {
        size_t l = strlen(e->d_name);
        if (l < 5 || l - 4 >= FACEID_NAME || strcmp(e->d_name + l - 4, ".emb")) continue;
        faceid_person *q = &g->p[g->people];
        memset(q->name, 0, sizeof q->name);
        memcpy(q->name, e->d_name, l - 4);
        if (!name_ok(q->name)) continue;
        snprintf(p, sizeof p, "%s/%s", g->dir, e->d_name);
        if (read_person(p, q) == 0 && q->count > 0) g->people++;
    }
    closedir(d);
    return 0;
}

int faceid_refresh(faceid_gallery *g) {
    if (dir_stamp(g->dir) == g->stamp) return 0;
    char dir[512];
    snprintf(dir, sizeof dir, "%s", g->dir);
    faceid_load(g, dir);
    return 1;
}

int faceid_save(const faceid_gallery *g, const faceid_person *p) {
    if (!name_ok(p->name)) return -1;
    mkdir(g->dir, 0700);
    char path[1024], tmp[1100];
    snprintf(path, sizeof path, "%s/%s.emb", g->dir, p->name);
    snprintf(tmp, sizeof tmp, "%s.new", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    uint32_t dim = FACEID_DIM, count = (uint32_t) p->count;
    int ok = fwrite("FID1", 1, 4, f) == 4 && fwrite(&dim, 4, 1, f) == 1 && fwrite(&count, 4, 1, f) == 1 &&
             fwrite(p->s, sizeof(float) * FACEID_DIM, count, f) == count;
    if (fclose(f) != 0) ok = 0;
    if (!ok || rename(tmp, path) < 0) { unlink(tmp); return -1; }
    return 0;
}

int faceid_forget(faceid_gallery *g, const char *name) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s.emb", g->dir, name);
    int r = unlink(path);
    for (int i = 0; i < g->people; i++)
        if (!strcmp(g->p[i].name, name)) {
            memmove(&g->p[i], &g->p[i + 1], sizeof g->p[0] * (size_t) (g->people - i - 1));
            g->people--;
            break;
        }
    return r;
}

faceid_person *faceid_person_get(faceid_gallery *g, const char *name) {
    if (!name_ok(name)) return NULL;
    for (int i = 0; i < g->people; i++)
        if (!strcmp(g->p[i].name, name)) return &g->p[i];
    if (g->people == FACEID_PEOPLE) return NULL;
    faceid_person *p = &g->p[g->people++];
    memset(p, 0, sizeof *p);
    snprintf(p->name, sizeof p->name, "%s", name);
    return p;
}

void faceid_person_add(faceid_person *p, const float e[FACEID_DIM]) {
    if (p->count == FACEID_SAMPLES) {
        memmove(p->s[0], p->s[1], sizeof p->s[0] * (FACEID_SAMPLES - 1));
        p->count--;
    }
    memcpy(p->s[p->count], e, sizeof p->s[0]);
    faceid_normalize(p->s[p->count]);
    p->count++;
}

const char *faceid_match(const faceid_gallery *g, const float e[FACEID_DIM], float *score) {
    const char *who = NULL;
    float best = -1;
    for (int i = 0; i < g->people; i++)
        for (int k = 0; k < g->p[i].count; k++) {
            float c = faceid_cosine(e, g->p[i].s[k]);
            if (c > best) { best = c; who = g->p[i].name; }
        }
    if (score) *score = best;
    return best >= FACEID_THRESHOLD ? who : NULL;
}

/* $XDG_RUNTIME_DIR/faceid/CAM.sample: "FIS2", the time (int64, us), the
 * faces in view (int32), the locked face the largest (int32), then the
 * embedding. */
static void sample_path(const char *cam, char *buf, size_t len, int make) {
    const char *r = getenv("XDG_RUNTIME_DIR");
    char d[512];
    snprintf(d, sizeof d, "%s/faceid", r && *r ? r : "/tmp");
    if (make) mkdir(d, 0700);
    snprintf(buf, len, "%s/%s.sample", d, cam);
}

int faceid_publish(const char *cam, const float e[FACEID_DIM], long long when_us, int faces, int largest) {
    char path[600], tmp[620];
    sample_path(cam, path, sizeof path, 1);
    snprintf(tmp, sizeof tmp, "%s.new", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    int32_t n = faces, big = largest;
    int64_t t = when_us;
    int ok = fwrite("FIS2", 1, 4, f) == 4 && fwrite(&t, 8, 1, f) == 1 && fwrite(&n, 4, 1, f) == 1 && fwrite(&big, 4, 1, f) == 1 &&
             fwrite(e, sizeof(float), FACEID_DIM, f) == FACEID_DIM;
    if (fclose(f) != 0) ok = 0;
    if (!ok || rename(tmp, path) < 0) { unlink(tmp); return -1; }
    return 0;
}

int faceid_read_sample(const char *cam, float e[FACEID_DIM], long long *when_us, int *faces, int *largest) {
    char path[600];
    sample_path(cam, path, sizeof path, 0);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4];
    int32_t n = 0;
    int64_t t = 0;
    int32_t big = 0;
    int ok = fread(magic, 1, 4, f) == 4 && !memcmp(magic, "FIS2", 4) && fread(&t, 8, 1, f) == 1 && fread(&n, 4, 1, f) == 1 &&
             fread(&big, 4, 1, f) == 1 && fread(e, sizeof(float), FACEID_DIM, f) == FACEID_DIM;
    fclose(f);
    if (!ok) return -1;
    *when_us = t;
    *faces = n;
    *largest = big;
    return 0;
}

int faceid_allowed(const char *list, const char *name) {
    if (!list || !name) return 0;
    size_t n = strlen(name);
    for (const char *p = list; *p;) {
        while (*p == ',' || *p == ' ') p++;
        const char *q = p;
        while (*q && *q != ',' && *q != ' ') q++;
        if ((size_t) (q - p) == n && !strncmp(p, name, n)) return 1;
        p = q;
    }
    return 0;
}
