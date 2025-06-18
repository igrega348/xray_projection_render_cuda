# X-ray Projection Renderer (CUDA Version)

This is a CUDA-accelerated version of the X-ray projection renderer, which generates X-ray projections of simple objects. The renderer supports various geometric primitives and deformation fields, with parallel processing on NVIDIA GPUs.
See repository https://www.github.com/igrega348/xray_projection_render/ for parent Go code.

## Features

- CUDA-accelerated ray integration
- Support for basic geometric primitives:
  - Sphere
  - Cylinder
  - Cube
- Deformation field support:
  - Linear deformation
  - Rigid deformation
  - Sigmoid deformation
  - Gaussian deformation
- Parallel processing of pixels using CUDA
- JSON/YAML input file support
- Configurable parameters:
  - Resolution
  - Number of projections
  - Field of view
  - Integration step size
  - Density multiplier
- Volume grid export
- Transforms.json generation for NeRF compatibility

## Requirements

- CUDA Toolkit (version 11.0 or later)
- CMake (version 3.8 or later)
- C++ compiler with C++14 support
- jsoncpp library
- NVIDIA GPU with compute capability 3.0 or higher

## Building

1. Create a build directory:
```bash
mkdir build
cd build
```

2. Configure with CMake:
```bash
cmake ..
```

3. Build the project:
```bash
cmake --build .
```

## Usage

The program can be run with the following command:

```bash
./xray_projection_render_cuda --input input.json --output output_dir [options]
```

### Command Line Options

- `--input`: Input JSON/YAML file describing the objects (required)
- `--output`: Output directory for rendered images (required)
- `--resolution`: Resolution of output images (default: 512)
- `--num_projections`: Number of projections to generate (default: 1)
- `--fov`: Field of view in degrees (default: 45.0)
- `--ds`: Integration step size (default: 0.01)
- `--density_multiplier`: Multiply all densities by this value (default: 1.0)
- `--export_volume`: Export voxel grid (default: false)

### Input File Format

The input file should be in JSON or YAML format. Here's an example with all supported object and deformation types:

```json
{
    "objects": [
        {
            "type": "sphere",
            "center": [0.0, 0.0, 0.0],
            "radius": 1.0,
            "rho": 1.0
        },
        {
            "type": "cube",
            "center": [0.0, 0.0, 0.0],
            "side": 2.0,
            "rho": -1.0
        },
        {
            "type": "cylinder",
            "p0": [0.0, 0.0, -1.0],
            "p1": [0.0, 0.0, 1.0],
            "radius": 0.5,
            "rho": 1.0
        }
    ],
    "deformations": [
        {
            "type": "linear",
            "exx": 0.1,
            "eyy": 0.1,
            "ezz": 0.1,
            "eyz": 0.0,
            "exz": 0.0,
            "exy": 0.0
        },
        {
            "type": "rigid",
            "ux": 0.1,
            "uy": 0.2,
            "uz": 0.3
        },
        {
            "type": "sigmoid",
            "amplitude": 0.5,
            "center": 0.0,
            "lengthscale": 0.2,
            "direction": "z"
        },
        {
            "type": "gaussian",
            "amplitudes": [0.1, 0.2, 0.3],
            "sigmas": [0.5, 0.5, 0.5],
            "centers": [0.0, 0.0, 0.0]
        }
    ]
}
```

### Deformation Types

1. Linear Deformation:
   - Parameters: exx, eyy, ezz, eyz, exz, exy (strain components)
   - Applies linear strain transformation to coordinates

2. Rigid Deformation:
   - Parameters: ux, uy, uz (displacement components)
   - Applies constant displacement to coordinates

3. Sigmoid Deformation:
   - Parameters:
     - amplitude: Maximum displacement
     - center: Center point of sigmoid
     - lengthscale: Controls the steepness
     - direction: 'x', 'y', or 'z' axis
   - Applies sigmoid function along specified axis

4. Gaussian Deformation:
   - Parameters:
     - amplitudes: [ax, ay, az] - Maximum displacement in each direction
     - sigmas: [sx, sy, sz] - Width of Gaussian in each direction
     - centers: [cx, cy, cz] - Center point of Gaussian
   - Applies Gaussian displacement field

## Performance

The CUDA implementation provides significant speedup compared to the CPU version, especially for:
- High-resolution images
- Multiple projections
- Complex scenes with many objects
- Small integration step sizes

## Output Files

1. Projection Images:
   - PNG format
   - Grayscale
   - Named as "projection_XXX.png" where XXX is the projection number

2. Transforms File:
   - JSON format
   - Contains camera parameters and transformation matrices
   - Compatible with NeRF framework

3. Volume Grid (if --export_volume is set):
   - Raw binary format
   - Float values
   - Resolution x Resolution x Resolution grid
   - ZXY ordering

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
