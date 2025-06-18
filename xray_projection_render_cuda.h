#pragma once

#include <vector>
#include <string>

// Function to save image using stb_image_write
void save_image(const std::vector<float>& data, int width, int height, const std::string& filename);

// CUDA kernel for computing volume slice
__global__ void compute_volume_slice_kernel(
    float* volume,
    int resolution,
    int z_slice
); 