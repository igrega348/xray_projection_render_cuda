
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <vector>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define flat_field (0.0f)
// const float flat_field = 0.0f;

// Structure for 3D vectors
struct Vec3 {
    float x, y, z;
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 normalize() const {
        float len = sqrtf(x*x + y*y + z*z);
        return Vec3(x/len, y/len, z/len);
    }

    __host__ __device__ Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    __host__ __device__ Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

    __host__ __device__ Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    __host__ __device__ float operator*(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    __host__ __device__ float length() const {
        return sqrtf(x*x + y*y + z*z);
    }

    __host__ __device__ Vec3 unit() const {
        return *this * (1.0f / length());
    }

    __host__ __device__ Vec3 cross(const Vec3& other) const {
        return Vec3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }

};

__host__ __device__ float density(const Vec3 x){
  const Vec3 c(0.0f, 0.0f, 0.0f);
  const float R = 1.0f;
  Vec3 dx = x - c;
  float r = dx.length();
  if (dx.length() < R){
    return 1.0f;
  }
  return 0.0f;
}

// Integrate the density along the ray from the origin to the end point.
// Hierarchical integration method which is more efficient than simple integration.
// Refines the integration step size based on the density of the scene.
__host__ __device__ float integrate_hierarchical(const Vec3 origin, const Vec3 _direction, float DS, float smin, float smax){
	Vec3 direction = _direction.normalize();
	// integrate using sliding window
	float right = smin + DS;
	float left = smin;
	float ds = DS / 10.0f;
	float prev_rho = 0.0f;
	float T = 0.0f; //flat_field;
	while (right <= smax){
    Vec3 x = origin + direction * right;
		float rho = density(x);
		if ((rho == 0) != (prev_rho == 0)){ // rho changed between left and right
			left += ds;
			while (left < right){
        x = origin + direction * left;
				T += density(x) * ds;
				left += ds;
			}
			T += rho * ds; // reuse rho from right
		} else {
			T += rho * DS;
		}
		prev_rho = rho;
		left = right;
		right += DS;
	}
	return exp(-T);
}

__global__ void assemble_image_kernel(
  float* output,
  int width
){
  int idx = blockIdx.x;
  if (idx>=width) return;

  // integrate over y
  float y = 2.0f*((float) idx) / ((float) width) - 1.0f;
  // float z = 2.0f*((float) idy) / ((float) height) - 1.0f;
  Vec3 x0(-3.0, y, 0.0);
  Vec3 x1(3.0, y, 0.0);
  Vec3 t = x1 - x0;
  output[idx] = integrate_hierarchical(x0, t, 0.01, 0.0, 6.0);
  return;
}

int main(){
  const int N = 20;
  // float *Ts;
  float *Ts = new float[N];
  cudaMallocManaged(&Ts, N*sizeof(float));
  std::cout << "Memory allocated" << std::endl;
  assemble_image_kernel<<<N,1>>>(Ts, N);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  std::cout << "Integral: " << std::endl;
  for (int i = 0; i<N; i++){
    std::cout << Ts[i] << std::endl;
  }
  std::cout << std::endl;
  // Free memory
  cudaFree(Ts);
  return 0;
}
