cmake_minimum_required(VERSION 3.8)

# Set CUDA paths explicitly
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc.exe")
set(CMAKE_CUDA_ARCHITECTURES 75)  # Adjust based on your GPU architecture

# Set Visual Studio paths
set(MSVC_PATH "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.36.32532")
set(MSVC_INCLUDE "${MSVC_PATH}/include")
set(MSVC_LIB "${MSVC_PATH}/lib/x64")

project(xray_projection_render_cuda CUDA CXX)

# Set CUDA specific settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include FetchContent for downloading dependencies
include(FetchContent)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(jsoncpp REQUIRED)

# Declare and make available argparse
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v2.9
)
FetchContent_MakeAvailable(argparse)

# Declare and make available stb_image_write
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)
FetchContent_MakeAvailable(stb)

# Add executable
add_executable(xray_projection_render_cuda 
    xray_projection_render_cuda.cu
    xray_projection_render_cuda.h
)

# Link libraries
target_link_libraries(xray_projection_render_cuda
    PRIVATE
    jsoncpp
    argparse
)

# Include directories
target_include_directories(xray_projection_render_cuda
    PRIVATE
    "${CUDA_TOOLKIT_ROOT_DIR}/include"
    "${JSONCPP_INCLUDE_DIRS}"
    "${stb_SOURCE_DIR}"
    "${MSVC_INCLUDE}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

# Set CUDA specific properties
set_target_properties(xray_projection_render_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
) 