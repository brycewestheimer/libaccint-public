# GPU Setup and Troubleshooting Guide

## Version: 0.1.0-alpha.2

This guide covers setting up GPU acceleration with LibAccInt for NVIDIA (CUDA) GPUs.

---

## Quick Start

### NVIDIA GPU (CUDA)

```bash
# Prerequisites: CUDA Toolkit 12.0+
# Verify CUDA installation
nvcc --version
nvidia-smi

# Build with CUDA
cmake --preset cuda-release
cmake --build --preset cuda-release

# Verify GPU detection
ctest --test-dir build/cuda-release -R test_backend
```

---

## Build Configuration

### CMake Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `LIBACCINT_USE_CUDA` | `AUTO`, `ON`, `OFF` | `AUTO` | CUDA backend |
| `LIBACCINT_ENABLE_EXPERIMENTAL_GPU_HESSIAN` | `ON`, `OFF` | `OFF` | Enable experimental GPU Hessian paths |
| `LIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES` | `ON`, `OFF` | `OFF` | Enable experimental GPU momentum/multipole property paths |

With `AUTO`, CMake detects GPU toolkits automatically. Use `ON` to require GPU support (fails if not found) or `OFF` to disable.
For v1 release, GPU Hessian and GPU property kernels are explicitly deferred and remain disabled by default.

### Manual CMake Configuration

```bash
# Explicit CUDA with specific toolkit path
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBACCINT_USE_CUDA=ON \
    -DCUDAToolkit_ROOT=/usr/local/cuda-12.0 \
    -DLIBACCINT_ENABLE_EXPERIMENTAL_GPU_HESSIAN=OFF \
    -DLIBACCINT_ENABLE_EXPERIMENTAL_GPU_PROPERTIES=OFF
```

### CUDA Architectures

LibAccInt targets common GPU architectures. Set `CMAKE_CUDA_ARCHITECTURES` for your GPU:

| GPU | Architecture | Flag |
|-----|-------------|------|
| V100 | sm_70 | `-DCMAKE_CUDA_ARCHITECTURES=70` |
| A100 | sm_80 | `-DCMAKE_CUDA_ARCHITECTURES=80` |
| H100 | sm_90 | `-DCMAKE_CUDA_ARCHITECTURES=90` |
| Multiple | sm_70;sm_80;sm_90 | `-DCMAKE_CUDA_ARCHITECTURES="70;80;90"` |

---

## Using GPU in Your Code

### Automatic Dispatch

The Engine automatically routes work to the GPU when beneficial:

```cpp
#include <libaccint/libaccint.hpp>

Engine engine(basis);

// Check GPU availability
if (engine.gpu_available()) {
    std::cout << "GPU backend is active\n";
}

// Automatic dispatch (GPU used when beneficial)
engine.compute(Operator::coulomb(), fock);  // May use GPU
```

### Explicit GPU Control

```cpp
// Force GPU for large workloads
engine.compute(Operator::coulomb(), fock, BackendHint::PreferGPU);

// Force CPU for debugging or comparison
engine.compute(Operator::coulomb(), fock, BackendHint::ForceCPU);
```

### Checking Backend at Runtime

```cpp
#include <libaccint/config.hpp>

std::cout << "CUDA: " << (has_cuda_backend() ? "yes" : "no") << "\n";
std::cout << "GPU available: " << engine.gpu_available() << "\n";
```

---

## Performance Tuning

### Dispatch Policy

Configure when GPU is preferred over CPU:

```cpp
DispatchConfig config;
config.gpu_threshold_shells = 10;       // Min shells for GPU dispatch
config.gpu_threshold_primitives = 100;  // Min primitives for GPU dispatch
Engine engine(basis, config);
```

### Memory Limits

```cpp
DispatchConfig config;
config.gpu_memory_limit = 2UL * 1024 * 1024 * 1024;  // 2 GB limit
Engine engine(basis, config);
```

---

## Troubleshooting

### Build Issues

#### "CUDA not found" with AUTO mode
```
-- CUDA backend: OFF (not found)
```
**Solution**: Ensure CUDA toolkit is in PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### "nvcc fatal: Unsupported gpu architecture"
**Solution**: Set the correct architecture for your GPU:
```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80  # For A100
```

### Runtime Issues

#### GPU detected but slower than CPU
**Likely cause**: Problem is too small for GPU overhead.
**Solution**: Use `BackendHint::Auto` (default) which makes this decision automatically, or increase the GPU threshold in DispatchConfig.

#### CUDA out of memory
```
CUDA error: out of memory
```
**Solutions**:
1. Reduce `config.gpu_memory_limit`
2. Use density fitting (reduces memory footprint)
3. Use a GPU with more memory
4. The Engine will fall back to CPU automatically

#### GPU compute results differ from CPU
Expected behavior: results should agree to ~1e-12. If differences are larger:
1. Check if using single precision (`MixedPrecisionFockBuilder`)
2. File a bug report with the system geometry and basis set

#### Multi-GPU not working
Ensure all GPUs are visible:
```bash
echo $CUDA_VISIBLE_DEVICES  # Should show all GPUs or be unset
nvidia-smi                    # Verify all GPUs appear
```

### Docker / Container Setup

```dockerfile
# CUDA container
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y cmake g++
COPY . /libaccint
WORKDIR /libaccint
RUN cmake --preset cuda-release && cmake --build --preset cuda-release
```

Run with GPU access:
```bash
docker run --gpus all my-libaccint-image
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0,1` |
| `LIBACCINT_GPU_MEMORY_LIMIT` | Override memory limit | `4096` (MB) |
| `OMP_NUM_THREADS` | CPU thread count | `8` |

---

## Verification

Run the GPU test suite after installation:

```bash
ctest --test-dir build/cuda-release -R "gpu|cuda" --output-on-failure
```

Expected: All GPU tests pass. If building CPU-only, GPU tests are automatically skipped.
