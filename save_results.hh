#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

inline void saveAssignmentsCSV(const std::vector<int>& assignments,
                                const std::string& filepath)
{
    std::ofstream out(filepath);
    if (!out) { std::cerr << "Failed to open: " << filepath << std::endl; return; }
    out << "point_idx,cluster\n";
    for (int i = 0; i < (int)assignments.size(); ++i)
        out << i << "," << assignments[i] << "\n";
    std::cout << "Saved: " << filepath << std::endl;
}

inline void saveCentroidsCSV(const std::vector<float>& centroids,
                              int K, int D,
                              const std::string& filepath)
{
    std::ofstream out(filepath);
    if (!out) { std::cerr << "Failed to open: " << filepath << std::endl; return; }
    for (int d = 0; d < D; ++d) {
        out << "dim_" << d;
        if (d < D - 1) out << ",";
    }
    out << "\n";
    for (int k = 0; k < K; ++k) {
        for (int d = 0; d < D; ++d) {
            out << centroids[k * D + d];
            if (d < D - 1) out << ",";
        }
        out << "\n";
    }
    std::cout << "Saved: " << filepath << std::endl;
}

inline void saveAssignmentsBin(const std::vector<int>& assignments,
                                const std::string& filepath)
{
    std::ofstream out(filepath, std::ios::binary);
    if (!out) { std::cerr << "Failed to open: " << filepath << std::endl; return; }
    int N = (int)assignments.size();
    out.write(reinterpret_cast<const char*>(&N), sizeof(int));
    out.write(reinterpret_cast<const char*>(assignments.data()),
              static_cast<size_t>(N) * sizeof(int));
    std::cout << "Saved: " << filepath << std::endl;
}

inline void saveCentroidsBin(const std::vector<float>& centroids,
                              int K, int D,
                              const std::string& filepath)
{
    std::ofstream out(filepath, std::ios::binary);
    if (!out) { std::cerr << "Failed to open: " << filepath << std::endl; return; }
    out.write(reinterpret_cast<const char*>(&K), sizeof(int));
    out.write(reinterpret_cast<const char*>(&D), sizeof(int));
    out.write(reinterpret_cast<const char*>(centroids.data()),
              static_cast<size_t>(K) * D * sizeof(float));
    std::cout << "Saved: " << filepath << std::endl;
}