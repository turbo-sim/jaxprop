/*
Minimal tester for jaxprop .cbin + generated _layout.h

Compile example (Windows, MSVC):
  cl /EHsc /W4 /std:c11 /DJX_LAYOUT_HEADER=\"CO2_400x400_layout.h\" test_cbin_reader.c

Compile example (gcc/clang):
  gcc -O2 -Wall -Wextra -std=c11 -DJX_LAYOUT_HEADER=\"CO2_400x400_layout.h\" test_cbin_reader.c -o test_cbin_reader

Run:
  test_cbin_reader CO2_400x400.cbin
*/

#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef JX_LAYOUT_HEADER
#define JX_LAYOUT_HEADER "CO2_400x400_layout.h"
#endif

#include JX_LAYOUT_HEADER

#if defined(_WIN32)
#define FSEEK _fseeki64
#define FTELL _ftelli64
#else
#define FSEEK fseeko
#define FTELL ftello
#endif

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t endian_tag;
    uint32_t nh;
    uint32_t np;
    uint32_t n_props;
    uint32_t n_node_fields;
    uint32_t n_coeff;
    uint32_t reserved;
    uint64_t phase_mask_offset;
    uint64_t provenance_offset;
    uint64_t provenance_size;
} jx_file_header_t;

typedef struct {
    double h_min;
    double h_max;
    double p_min;
    double p_max;
    double delta_h;
    double delta_logp;
} jx_meta_t;

_Static_assert(sizeof(jx_file_header_t) == 64, "The v2 file header requires a 64-byte layout");
_Static_assert(sizeof(jx_prop_record_t) == 120, "Property directory records require a 120-byte layout");

static long long file_size_bytes(FILE *fp) {
    if (FSEEK(fp, 0, SEEK_END) != 0) {
        return -1;
    }
    long long n = (long long)FTELL(fp);
    if (n < 0) {
        return -1;
    }
    if (FSEEK(fp, 0, SEEK_SET) != 0) {
        return -1;
    }
    return n;
}

static int read_double_at(FILE *fp, uint64_t base_offset, size_t flat_index, double *out) {
    uint64_t abs_off = base_offset + (uint64_t)flat_index * (uint64_t)sizeof(double);
    if (FSEEK(fp, (long long)abs_off, SEEK_SET) != 0) {
        return -1;
    }
    return (fread(out, sizeof(double), 1, fp) == 1) ? 0 : -1;
}

static const jx_prop_record_t *find_prop(const jx_prop_record_t *recs, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; ++i) {
        if (strncmp(recs[i].name, name, sizeof(recs[i].name)) == 0) {
            return &recs[i];
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    const char *cbin_path = (argc > 1) ? argv[1] : "CO2_400x400.cbin";

    FILE *fp = fopen(cbin_path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s (%s)\n", cbin_path, strerror(errno));
        return 1;
    }

    long long nbytes = file_size_bytes(fp);
    if (nbytes < 0) {
        fprintf(stderr, "Error: cannot determine file size\n");
        fclose(fp);
        return 1;
    }

    jx_file_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
        fprintf(stderr, "Error: failed to read file header\n");
        fclose(fp);
        return 1;
    }

    if (strncmp(hdr.magic, JX_MAGIC_EXPECTED, 7) != 0) {
        fprintf(stderr, "Error: magic mismatch. Got '%.8s' expected '%s'\n", hdr.magic, JX_MAGIC_EXPECTED);
        fclose(fp);
        return 1;
    }
    if (hdr.version != JX_VERSION_EXPECTED) {
        fprintf(stderr, "Error: version mismatch. Got %u expected %u\n", hdr.version, JX_VERSION_EXPECTED);
        fclose(fp);
        return 1;
    }
    if (hdr.endian_tag != 0x01020304u || hdr.nh < 2 || hdr.np < 2 ||
        hdr.provenance_offset + hdr.provenance_size != (uint64_t)nbytes ||
        hdr.phase_mask_offset + (uint64_t)(hdr.nh - 1) * (hdr.np - 1) != hdr.provenance_offset) {
        fprintf(stderr, "Error: unsupported endian or inconsistent v2 payload offsets\n");
        fclose(fp);
        return 1;
    }
    if (hdr.n_node_fields != JX_NODE_FIELDS || hdr.n_coeff != JX_COEFFS_PER_CELL) {
        fprintf(stderr, "Error: layout mismatch. node_fields=%u coeff=%u (expected %u/%u)\n",
                hdr.n_node_fields, hdr.n_coeff, JX_NODE_FIELDS, JX_COEFFS_PER_CELL);
        fclose(fp);
        return 1;
    }
    if (hdr.n_props == 0 ||
        (uint64_t)sizeof(jx_file_header_t) + sizeof(jx_meta_t) +
        (uint64_t)hdr.n_props * sizeof(jx_prop_record_t) > hdr.phase_mask_offset) {
        fprintf(stderr, "Error: invalid property directory extent\n");
        fclose(fp);
        return 1;
    }

    jx_meta_t meta;
    if (fread(&meta, sizeof(meta), 1, fp) != 1) {
        fprintf(stderr, "Error: failed to read metadata\n");
        fclose(fp);
        return 1;
    }

    jx_prop_record_t *recs = (jx_prop_record_t *)malloc((size_t)hdr.n_props * sizeof(jx_prop_record_t));
    if (!recs) {
        fprintf(stderr, "Error: malloc failed for property records\n");
        fclose(fp);
        return 1;
    }

    if (fread(recs, sizeof(jx_prop_record_t), hdr.n_props, fp) != hdr.n_props) {
        fprintf(stderr, "Error: failed to read property records\n");
        free(recs);
        fclose(fp);
        return 1;
    }

    printf("Read file: %s\n", cbin_path);
    printf("Size: %lld bytes\n", nbytes);
    printf("Grid: Nh=%u, Np=%u\n", hdr.nh, hdr.np);
    printf("Properties: %u\n", hdr.n_props);
    printf("Metadata: h=[%.6e, %.6e], p=[%.6e, %.6e], dh=%.6e, dlogP=%.6e\n",
           meta.h_min, meta.h_max, meta.p_min, meta.p_max, meta.delta_h, meta.delta_logp);

    /* Pick pressure if available, else first property */
    const jx_prop_record_t *prop = find_prop(recs, hdr.n_props, "pressure");
    if (!prop) {
        prop = &recs[0];
    }

    printf("Testing property: %s\n", prop->name);

    uint32_t i = hdr.nh / 2;
    uint32_t j = hdr.np / 2;
    size_t node_idx = JX_NODE_IDX(i, j, hdr.np);

    uint32_t ci = (hdr.nh > 1) ? (hdr.nh - 2) / 2 : 0;
    uint32_t cj = (hdr.np > 1) ? (hdr.np - 2) / 2 : 0;
    uint32_t k = 7;
    size_t coeff_idx = JX_COEFF_IDX(ci, cj, k, hdr.np - 1);
    uint8_t cell_valid = 0;
    if (FSEEK(fp, (long long)(hdr.phase_mask_offset + JX_CELL_IDX(ci, cj, hdr.np - 1)), SEEK_SET) != 0 ||
        fread(&cell_valid, sizeof(cell_valid), 1, fp) != 1 || cell_valid > 1) {
        fprintf(stderr, "Error: invalid phase-cell validity payload\n");
        free(recs);
        fclose(fp);
        return 1;
    }
    printf("Cell (%u,%u) eligible: %u (never interpolate if zero)\n", ci, cj, (unsigned)cell_valid);
    printf("Complete JSON provenance/saturation payload: %llu bytes\n", (unsigned long long)hdr.provenance_size);

    double v = NAN, gh = NAN, glp = NAN, c7 = NAN;

    if (read_double_at(fp, prop->off_value, node_idx, &v) != 0 ||
        read_double_at(fp, prop->off_grad_h, node_idx, &gh) != 0 ||
        read_double_at(fp, prop->off_grad_logP, node_idx, &glp) != 0 ||
        read_double_at(fp, prop->off_coeffs, coeff_idx, &c7) != 0) {
        fprintf(stderr, "Error: failed to read sample values using offsets/indexing\n");
        free(recs);
        fclose(fp);
        return 1;
    }

    printf("Sample node (i=%u, j=%u): value=%.9e, grad_h=%.9e, grad_logP=%.9e\n", i, j, v, gh, glp);
    printf("Sample coeff (cell_i=%u, cell_j=%u, k=%u): %.9e\n", ci, cj, k, c7);
    /* Machine-readable cross-language regression record (17 digits round-trip). */
    printf("JX_SAMPLE {\"property\":\"%s\",\"node_i\":%u,\"node_j\":%u,\"cell_i\":%u,\"cell_j\":%u,\"k\":%u,\"cell_valid\":%u,\"value\":%.17g,\"grad_h\":%.17g,\"grad_logP\":%.17g,\"coefficient\":%.17g}\n",
           prop->name, i, j, ci, cj, k, (unsigned)cell_valid, v, gh, glp, c7);

    /* Print first few property names */
    printf("First properties in directory:\n");
    uint32_t show_n = (hdr.n_props < 8u) ? hdr.n_props : 8u;
    for (uint32_t n = 0; n < show_n; ++n) {
        printf("  [%u] %s\n", n, recs[n].name);
    }

    free(recs);
    fclose(fp);
    return 0;
}


