#ifndef SBIN_HH
#define SBIN_HH

#include <iostream>
#include <string>
#include <cstdint>

// SBIN File Header
struct SbinHeader {
    char magic[4];
    uint32_t version;
    uint64_t n;
    uint32_t d;
    uint32_t k_meta;
    uint32_t dtype_code;
    char reserved[36];
};

// Function declaration for loading SBIN payload in Structure of Arrays (SoA) format
bool loadSbinSoA(const std::string& path, float* h_points_SoA, int N, int D);

#endif // SBIN_HH