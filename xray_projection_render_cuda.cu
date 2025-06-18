#define _USE_MATH_DEFINES  // For M_PI
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <json/json.h>
#include <filesystem>
#include <argparse/argparse.hpp>
#include <chrono>
#include <iomanip>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <windows.h>  // For Windows-specific functions
#include "xray_projection_render_cuda.h"  // Include our header

// Constants
constexpr float PI = 3.14159265358979323846f;
constexpr float CUBE_HALF_DIAGONAL = 1.74f;

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
};

// Base class for objects
class Object {
public:
    virtual __device__ float density(float x, float y, float z) const = 0;
    virtual ~Object() = default;
};

// Sphere object
class Sphere : public Object {
private:
    Vec3 center;
    float radius;
    float rho;
    
public:
    __host__ __device__ Sphere(const Vec3& center, float radius, float rho)
        : center(center), radius(radius), rho(rho) {}
    
    __device__ float density(float x, float y, float z) const override {
        float dx = x - center.x;
        float dy = y - center.y;
        float dz = z - center.z;
        float dist_sq = dx*dx + dy*dy + dz*dz;
        return (dist_sq <= radius*radius) ? rho : 0.0f;
    }
};

// Cylinder object
class Cylinder : public Object {
private:
    Vec3 p0, p1;
    float radius;
    float rho;
    
public:
    __host__ __device__ Cylinder(const Vec3& p0, const Vec3& p1, float radius, float rho)
        : p0(p0), p1(p1), radius(radius), rho(rho) {}
    
    __device__ float density(float x, float y, float z) const override {
        Vec3 axis = Vec3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
        Vec3 point = Vec3(x - p0.x, y - p0.y, z - p0.z);
        
        float axis_length_sq = axis.x*axis.x + axis.y*axis.y + axis.z*axis.z;
        float t = (point.x*axis.x + point.y*axis.y + point.z*axis.z) / axis_length_sq;
        
        if (t < 0.0f || t > 1.0f) return 0.0f;
        
        Vec3 projection = Vec3(
            p0.x + t*axis.x,
            p0.y + t*axis.y,
            p0.z + t*axis.z
        );
        
        float dx = x - projection.x;
        float dy = y - projection.y;
        float dz = z - projection.z;
        float dist_sq = dx*dx + dy*dy + dz*dz;
        
        return (dist_sq <= radius*radius) ? rho : 0.0f;
    }
};

// Cube object
class Cube : public Object {
private:
    Vec3 center;
    float side;
    float rho;
    
public:
    __host__ __device__ Cube(const Vec3& center, float side, float rho)
        : center(center), side(side), rho(rho) {}
    
    __device__ float density(float x, float y, float z) const override {
        float half_side = side * 0.5f;
        if (fabsf(x - center.x) <= half_side &&
            fabsf(y - center.y) <= half_side &&
            fabsf(z - center.z) <= half_side) {
            return rho;
        }
        return 0.0f;
    }
};

// Deformation base class
class Deformation {
public:
    virtual __device__ void apply(float& x, float& y, float& z) const = 0;
    virtual ~Deformation() = default;
};

// Linear deformation
class LinearDeformation : public Deformation {
private:
    float exx, eyy, ezz, eyz, exz, exy;
    
public:
    __host__ __device__ LinearDeformation(float exx, float eyy, float ezz,
                                        float eyz, float exz, float exy)
        : exx(exx), eyy(eyy), ezz(ezz), eyz(eyz), exz(exz), exy(exy) {}
    
    __device__ void apply(float& x, float& y, float& z) const override {
        float new_x = x + exx*x + exy*y + exz*z;
        float new_y = y + exy*x + eyy*y + eyz*z;
        float new_z = z + exz*x + eyz*y + ezz*z;
        x = new_x;
        y = new_y;
        z = new_z;
    }
};

// Rigid deformation
class RigidDeformation : public Deformation {
private:
    float ux, uy, uz;
    
public:
    __host__ __device__ RigidDeformation(float ux, float uy, float uz)
        : ux(ux), uy(uy), uz(uz) {}
    
    __device__ void apply(float& x, float& y, float& z) const override {
        x += ux;
        y += uy;
        z += uz;
    }
};

// Sigmoid deformation
class SigmoidDeformation : public Deformation {
private:
    float amplitude;
    float center;
    float lengthscale;
    char direction;  // 'x', 'y', or 'z'
    
public:
    __host__ __device__ SigmoidDeformation(float amplitude, float center, float lengthscale, char direction)
        : amplitude(amplitude), center(center), lengthscale(lengthscale), direction(direction) {}
    
    __device__ void apply(float& x, float& y, float& z) const override {
        switch (direction) {
            case 'x':
                x += amplitude / (1.0f + expf(-(x - center) / lengthscale));
                break;
            case 'y':
                y += amplitude / (1.0f + expf(-(y - center) / lengthscale));
                break;
            case 'z':
                z += amplitude / (1.0f + expf(-(z - center) / lengthscale));
                break;
        }
    }
};

// Gaussian deformation
class GaussianDeformation : public Deformation {
private:
    float amplitudes[3];
    float sigmas[3];
    float centers[3];
    
public:
    __host__ __device__ GaussianDeformation(
        float ax, float ay, float az,
        float sx, float sy, float sz,
        float cx, float cy, float cz
    ) {
        amplitudes[0] = ax;
        amplitudes[1] = ay;
        amplitudes[2] = az;
        sigmas[0] = sx;
        sigmas[1] = sy;
        sigmas[2] = sz;
        centers[0] = cx;
        centers[1] = cy;
        centers[2] = cz;
    }
    
    __device__ void apply(float& x, float& y, float& z) const override {
        float dx = x - centers[0];
        float dy = y - centers[1];
        float dz = z - centers[2];
        float r_sq = dx*dx + dy*dy + dz*dz;
        
        x += amplitudes[0] * expf(-r_sq / (2.0f * sigmas[0] * sigmas[0]));
        y += amplitudes[1] * expf(-r_sq / (2.0f * sigmas[1] * sigmas[1]));
        z += amplitudes[2] * expf(-r_sq / (2.0f * sigmas[2] * sigmas[2]));
    }
};

// Move global variables to device memory
__device__ Object** d_objects = nullptr;
__device__ int num_objects = 0;
__device__ Deformation** d_deformations = nullptr;
__device__ int num_deformations = 0;
__device__ float density_multiplier = 1.0f;
__device__ float flat_field = 0.0f;

// CUDA kernel for ray integration
__global__ void integrate_ray_kernel(
    float* output,
    int width,
    int height,
    Vec3 origin,
    Vec3 direction,
    float ds,
    float smin,
    float smax
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx >= width || idy >= height) return;
    
    // Calculate pixel position
    float x = origin.x + direction.x * smin;
    float y = origin.y + direction.y * smin;
    float z = origin.z + direction.z * smin;
    
    // Apply deformations
    for (int i = 0; i < num_deformations; i++) {
        d_deformations[i]->apply(x, y, z);
    }
    
    // Integrate along ray
    float T = flat_field;
    float s = smin;
    while (s < smax) {
        float rho = 0.0f;
        for (int i = 0; i < num_objects; i++) {
            rho += d_objects[i]->density(x, y, z);
        }
        T += rho * ds * density_multiplier;
        
        x += direction.x * ds;
        y += direction.y * ds;
        z += direction.z * ds;
        s += ds;
    }
    
    output[idy * width + idx] = expf(-T);
}

// Host function to initialize CUDA resources
void init_cuda() {
    // Allocate device memory for objects and deformations
    cudaMalloc(&d_objects, sizeof(Object*) * num_objects);
    cudaMalloc(&d_deformations, sizeof(Deformation*) * num_deformations);
}

// Host function to cleanup CUDA resources
void cleanup_cuda() {
    if (d_objects) cudaFree(d_objects);
    if (d_deformations) cudaFree(d_deformations);
}

// Replace isatty with Windows equivalent
inline bool check_is_terminal() {
    return GetFileType(GetStdHandle(STD_OUTPUT_HANDLE)) == FILE_TYPE_CHAR;
}

// Progress reporting class
class ProgressReporter {
private:
    int total;
    int current;
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    bool is_terminal;
    int bar_width;
    
public:
    ProgressReporter(int total_steps) : total(total_steps), current(0) {
        start_time = std::chrono::steady_clock::now();
        is_terminal = check_is_terminal();
        bar_width = 50;
    }
    
    void update(int step = 1) {
        current += step;
        if (is_terminal) {
            float progress = static_cast<float>(current) / total;
            int filled = static_cast<int>(bar_width * progress);
            
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            float rate = static_cast<float>(current) / elapsed;
            int remaining = static_cast<int>((total - current) / rate);
            
            std::cout << "\r[";
            for (int i = 0; i < bar_width; i++) {
                if (i < filled) std::cout << "=";
                else if (i == filled) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << std::fixed << std::setprecision(1) 
                      << (progress * 100.0f) << "% "
                      << current << "/" << total
                      << " (" << format_time(remaining) << " remaining)";
            std::cout.flush();
        } else {
            std::cout << "Progress: " << current << "/" << total << " (" 
                      << std::fixed << std::setprecision(1) 
                      << (static_cast<float>(current) / total * 100.0f) << "%)" << std::endl;
        }
    }
    
    void finish() {
        if (is_terminal) {
            std::cout << std::endl;
        }
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        std::cout << "Completed in " << format_time(elapsed) << std::endl;
    }
    
private:
    std::string format_time(int seconds) {
        int hours = seconds / 3600;
        int minutes = (seconds % 3600) / 60;
        int secs = seconds % 60;
        
        std::stringstream ss;
        if (hours > 0) ss << hours << "h ";
        if (minutes > 0 || hours > 0) ss << minutes << "m ";
        ss << secs << "s";
        return ss.str();
    }
};

// Main rendering function
void render(
    const std::string& input_file,
    const std::string& output_dir,
    int resolution,
    int num_projections,
    float fov,
    float ds,
    float density_mult,
    bool export_volume
) {
    std::cout << "Initializing CUDA..." << std::endl;
    init_cuda();
    
    // Set up camera parameters
    float aspect_ratio = 1.0f;
    float tan_fov = tanf(fov * 0.5f * PI / 180.0f);
    
    // Calculate grid and block dimensions
    dim3 block_dim(16, 16);
    dim3 grid_dim(
        (resolution + block_dim.x - 1) / block_dim.x,
        (resolution + block_dim.y - 1) / block_dim.y
    );
    
    // Allocate output buffer
    float* d_output;
    cudaMalloc(&d_output, resolution * resolution * sizeof(float));
    
    std::cout << "Rendering " << num_projections << " projections at " 
              << resolution << "x" << resolution << " resolution..." << std::endl;
    
    ProgressReporter progress(num_projections);
    
    // Render each projection
    for (int proj = 0; proj < num_projections; proj++) {
        float angle = 2.0f * PI * proj / num_projections;
        
        // Calculate camera position and direction
        Vec3 origin(cosf(angle), sinf(angle), 0.0f);
        Vec3 direction = Vec3(-cosf(angle), -sinf(angle), 0.0f).normalize();
        
        // Launch kernel
        integrate_ray_kernel<<<grid_dim, block_dim>>>(
            d_output,
            resolution,
            resolution,
            origin,
            direction,
            ds,
            -CUBE_HALF_DIAGONAL,
            CUBE_HALF_DIAGONAL
        );
        
        // Copy result back to host and save image
        std::vector<float> output(resolution * resolution);
        cudaMemcpy(output.data(), d_output, resolution * resolution * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Save image
        save_image(output, resolution, resolution, output_dir + "/projection_" + std::to_string(proj) + ".png");
        
        progress.update();
    }
    
    progress.finish();
    
    // Cleanup
    cudaFree(d_output);
    cleanup_cuda();
}

// Function to save image using stb_image_write
void save_image(const std::vector<float>& data, int width, int height, const std::string& filename) {
    std::vector<unsigned char> pixels(width * height);
    // Find max value in data
    float max_value = *std::max_element(data.begin(), data.end());
    std::cout << "Max value: " << max_value << std::endl;
    // Find min value in data
    float min_value = *std::min_element(data.begin(), data.end());
    std::cout << "Min value: " << min_value << std::endl;
    // Convert float data to 8-bit grayscale
    for (int i = 0; i < width * height; i++) {
        float value = data[i];
        // Normalize to [0, 255] range
        value = value / max_value;
        // // Clamp to [0, 1] range
        // value = fmaxf(0.0f, fminf(1.0f, value));
        // Convert to 8-bit
        pixels[i] = static_cast<unsigned char>(value * 255.0f);
    }
    
    // Create directory if it doesn't exist
    std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
    
    // Save image
    stbi_write_png(filename.c_str(), width, height, 1, pixels.data(), width);
}

// Function to save transforms.json
void save_transforms(const std::string& filename, int num_projections, float fov, int resolution) {
    Json::Value root;
    root["camera_angle_x"] = fov * M_PI / 180.0f;
    root["fl_x"] = resolution / (2.0f * tanf(fov * 0.5f * M_PI / 180.0f));
    root["fl_y"] = root["fl_x"];
    root["w"] = resolution;
    root["h"] = resolution;
    root["cx"] = resolution / 2.0f;
    root["cy"] = resolution / 2.0f;
    
    Json::Value frames(Json::arrayValue);
    for (int i = 0; i < num_projections; i++) {
        float angle = 2.0f * M_PI * i / num_projections;
        Json::Value frame;
        frame["file_path"] = "projection_" + std::to_string(i) + ".png";
        frame["time"] = static_cast<float>(i);
        
        // Create transform matrix (4x4)
        Json::Value transform(Json::arrayValue);
        for (int row = 0; row < 4; row++) {
            Json::Value row_array(Json::arrayValue);
            for (int col = 0; col < 4; col++) {
                if (row == col) {
                    row_array.append(1.0f);
                } else {
                    row_array.append(0.0f);
                }
            }
            transform.append(row_array);
        }
        
        // Set rotation around Z axis
        transform[0][0] = cosf(angle);
        transform[0][1] = -sinf(angle);
        transform[1][0] = sinf(angle);
        transform[1][1] = cosf(angle);
        
        frame["transform_matrix"] = transform;
        frames.append(frame);
    }
    root["frames"] = frames;
    
    // Write to file
    std::ofstream file(filename);
    Json::StyledWriter writer;
    file << writer.write(root);
}

// Function to load objects from JSON/YAML file
void load_objects(const std::string& filename) {
    std::ifstream file(filename);
    Json::Value root;
    Json::CharReaderBuilder reader;
    std::string errors;
    
    if (!Json::parseFromStream(reader, file, &root, &errors)) {
        throw std::runtime_error("Failed to parse input file: " + errors);
    }
    
    // Load objects
    const Json::Value& objects = root["objects"];
    num_objects = objects.size();
    
    // Allocate host memory for object pointers
    std::vector<Object*> h_objects(num_objects);
    
    for (int i = 0; i < num_objects; i++) {
        const Json::Value& obj = objects[i];
        std::string type = obj["type"].asString();
        
        if (type == "sphere") {
            const Json::Value& center = obj["center"];
            Vec3 center_vec(center[0].asFloat(), center[1].asFloat(), center[2].asFloat());
            h_objects[i] = new Sphere(center_vec, obj["radius"].asFloat(), obj["rho"].asFloat());
        }
        else if (type == "cube") {
            const Json::Value& center = obj["center"];
            Vec3 center_vec(center[0].asFloat(), center[1].asFloat(), center[2].asFloat());
            h_objects[i] = new Cube(center_vec, obj["side"].asFloat(), obj["rho"].asFloat());
        }
        else if (type == "cylinder") {
            const Json::Value& p0 = obj["p0"];
            const Json::Value& p1 = obj["p1"];
            Vec3 p0_vec(p0[0].asFloat(), p0[1].asFloat(), p0[2].asFloat());
            Vec3 p1_vec(p1[0].asFloat(), p1[1].asFloat(), p1[2].asFloat());
            h_objects[i] = new Cylinder(p0_vec, p1_vec, obj["radius"].asFloat(), obj["rho"].asFloat());
        }
    }
    
    // Copy objects to device
    cudaMemcpy(d_objects, h_objects.data(), sizeof(Object*) * num_objects, cudaMemcpyHostToDevice);
    
    // Load deformations
    const Json::Value& deformations = root["deformations"];
    num_deformations = deformations.size();
    
    if (num_deformations > 0) {
        std::vector<Deformation*> h_deformations(num_deformations);
        
        for (int i = 0; i < num_deformations; i++) {
            const Json::Value& def = deformations[i];
            std::string type = def["type"].asString();
            
            if (type == "linear") {
                h_deformations[i] = new LinearDeformation(
                    def["exx"].asFloat(),
                    def["eyy"].asFloat(),
                    def["ezz"].asFloat(),
                    def["eyz"].asFloat(),
                    def["exz"].asFloat(),
                    def["exy"].asFloat()
                );
            }
            else if (type == "rigid") {
                h_deformations[i] = new RigidDeformation(
                    def["ux"].asFloat(),
                    def["uy"].asFloat(),
                    def["uz"].asFloat()
                );
            }
            else if (type == "sigmoid") {
                h_deformations[i] = new SigmoidDeformation(
                    def["amplitude"].asFloat(),
                    def["center"].asFloat(),
                    def["lengthscale"].asFloat(),
                    def["direction"].asString()[0]
                );
            }
            else if (type == "gaussian") {
                h_deformations[i] = new GaussianDeformation(
                    def["amplitudes"][0].asFloat(),
                    def["amplitudes"][1].asFloat(),
                    def["amplitudes"][2].asFloat(),
                    def["sigmas"][0].asFloat(),
                    def["sigmas"][1].asFloat(),
                    def["sigmas"][2].asFloat(),
                    def["centers"][0].asFloat(),
                    def["centers"][1].asFloat(),
                    def["centers"][2].asFloat()
                );
            }
        }
        
        // Copy deformations to device
        cudaMemcpy(d_deformations, h_deformations.data(), sizeof(Deformation*) * num_deformations, cudaMemcpyHostToDevice);
    }
}

// Function to export volume grid
void export_volume(int resolution, const std::string& filename) {
    std::cout << "Exporting volume grid at " << resolution << "x" << resolution << "x" << resolution << " resolution..." << std::endl;
    
    std::vector<float> volume(resolution * resolution * resolution);
    
    // Calculate grid and block dimensions
    dim3 block_dim(8, 8, 8);
    dim3 grid_dim(
        (resolution + block_dim.x - 1) / block_dim.x,
        (resolution + block_dim.y - 1) / block_dim.y,
        (resolution + block_dim.z - 1) / block_dim.z
    );
    
    ProgressReporter progress(resolution);
    
    // Process volume in slices to show progress
    for (int z = 0; z < resolution; z++) {
        // Launch kernel for current slice
        compute_volume_slice_kernel<<<grid_dim, block_dim>>>(
            volume.data(),
            resolution,
            z
        );
        
        progress.update();
    }
    
    progress.finish();
    
    std::cout << "Saving volume to file..." << std::endl;
    // Save volume to file
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(volume.data()), volume.size() * sizeof(float));
    std::cout << "Volume saved to " << filename << std::endl;
}

// CUDA kernel for computing volume slice
__global__ void compute_volume_slice_kernel(
    float* volume,
    int resolution,
    int z_slice
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = z_slice;
    
    if (x >= resolution || y >= resolution) return;
    
    // Convert grid coordinates to world coordinates
    float world_x = (float)x / resolution * 2.0f - 1.0f;
    float world_y = (float)y / resolution * 2.0f - 1.0f;
    float world_z = (float)z / resolution * 2.0f - 1.0f;
    
    // Apply deformations
    for (int i = 0; i < num_deformations; i++) {
        d_deformations[i]->apply(world_x, world_y, world_z);
    }
    
    // Compute density
    float rho = 0.0f;
    for (int i = 0; i < num_objects; i++) {
        rho += d_objects[i]->density(world_x, world_y, world_z);
    }
    
    // Store in volume
    volume[z * resolution * resolution + y * resolution + x] = rho * density_multiplier;
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("xray_projection_render_cuda");
    
    program.add_argument("--input")
        .required()
        .help("Input JSON/YAML file describing the objects");
    
    program.add_argument("--output")
        .required()
        .help("Output directory for rendered images");
    
    program.add_argument("--resolution")
        .default_value(512)
        .scan<'i', int>()
        .help("Resolution of output images");
    
    program.add_argument("--num_projections")
        .default_value(1)
        .scan<'i', int>()
        .help("Number of projections to generate");
    
    program.add_argument("--fov")
        .default_value(45.0f)
        .scan<'f', float>()
        .help("Field of view in degrees");
    
    program.add_argument("--ds")
        .default_value(0.01f)
        .scan<'f', float>()
        .help("Integration step size");
    
    program.add_argument("--density_multiplier")
        .default_value(1.0f)
        .scan<'f', float>()
        .help("Multiply all densities by this value");
    
    program.add_argument("--flat_field")
        .default_value(0.0f)
        .scan<'f', float>()
        .help("Add a constant value to all pixels (flat field correction)");
    
    program.add_argument("--export_volume")
        .default_value(false)
        .implicit_value(true)
        .help("Export voxel grid");
    
    try {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }
    
    // Get arguments
    std::string input_file = program.get<std::string>("--input");
    std::string output_dir = program.get<std::string>("--output");
    int resolution = program.get<int>("--resolution");
    int num_projections = program.get<int>("--num_projections");
    float fov = program.get<float>("--fov");
    float ds = program.get<float>("--ds");
    float density_mult = program.get<float>("--density_multiplier");
    float flat_field_val = program.get<float>("--flat_field");
    bool export_volume_flag = program.get<bool>("--export_volume");
    
    try {
        std::cout << "Loading objects and deformations from " << input_file << "..." << std::endl;
        load_objects(input_file);
        std::cout << "Loaded " << num_objects << " objects and " << num_deformations << " deformations" << std::endl;
        
        // Set global parameters
        density_multiplier = density_mult;
        flat_field = flat_field_val;
        
        // Render projections
        render(
            input_file,
            output_dir,
            resolution,
            num_projections,
            fov,
            ds,
            density_mult,
            export_volume_flag
        );
        
        std::cout << "Saving transforms.json..." << std::endl;
        save_transforms(output_dir + "/transforms.json", num_projections, fov, resolution);
        
        // Export volume if requested
        if (export_volume_flag) {
            export_volume(resolution, output_dir + "/volume.raw");
        }
        
        std::cout << "All operations completed successfully!" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 