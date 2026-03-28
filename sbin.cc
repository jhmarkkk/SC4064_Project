#include "sbin.hh"

#include <cfloat>
#include <vector>
#include <random>
#include <fstream>
#include <cstring>
#include <iomanip>

bool loadSbinSoA(const std::string& path, float* h_points_SoA, int N, int D) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open SBIN file: " << path << std::endl;
        return false;
    }

    SbinHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in) {
        std::cerr << "Failed to read SBIN header.\n";
        return false;
    }
    if (std::memcmp(header.magic, "SBIN", 4) != 0 || header.version != 1 || header.dtype_code != 1) {
        std::cerr << "Unsupported SBIN header.\n";
        return false;
    }
    if (static_cast<int>(header.n) != N || static_cast<int>(header.d) != D) {
        std::cerr << "Header mismatch. Expected N=" << N << " D=" << D
                  << " but got N=" << header.n << " D=" << header.d << std::endl;
        return false;
    }

    in.read(reinterpret_cast<char*>(h_points_SoA), static_cast<size_t>(N) * D * sizeof(float));
    if (!in) {
        std::cerr << "Failed to read SBIN payload.\n";
        return false;
    }
    return true;
}