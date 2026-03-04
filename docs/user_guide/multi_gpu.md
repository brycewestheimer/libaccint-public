# Multi-GPU and Distributed Computing

> **Experimental:** Multi-GPU support is not part of the `0.1.0-alpha.2`
> guarantee. The alpha release is single-GPU focused. The APIs in this guide
> remain public for specialist users but may change without notice.

LibAccInt supports scaling molecular integral computations across multiple GPUs and distributed systems using MPI.

## Multi-GPU on Single Node

### Quick Start

```cpp
#include <libaccint/engine/multi_gpu_engine.hpp>
#include <libaccint/consumers/multi_gpu_fock_builder.hpp>

using namespace libaccint;

// Create basis set
BasisSet basis(shells);

// Configure multi-GPU execution
engine::MultiGPUConfig config;
config.device_ids = {0, 1, 2, 3};  // Use 4 GPUs
config.enable_peer_access = true;  // Enable NVLink/P2P where available

// Create multi-GPU engine
engine::MultiGPUEngine engine(basis, config);

// Create multi-GPU Fock builder
consumers::MultiGPUFockBuilder fock(basis.n_basis_functions(), config.device_ids);
fock.set_density(D.data(), nbf);

// Compute all two-electron integrals
engine.compute_all_eri(fock);

// Get results (automatically reduced from all GPUs)
auto J = fock.get_coulomb_matrix();
auto K = fock.get_exchange_matrix();
```

### Device Management

```cpp
#include <libaccint/device/device_manager.hpp>

using namespace libaccint::device;

// Get the device manager singleton
auto& dm = DeviceManager::instance();

// Query available devices
std::cout << "Available GPUs: " << dm.device_count() << "\n";

for (const auto& props : dm.available_devices()) {
    std::cout << "  [" << props.device_id << "] " << props.name 
              << " (" << props.multiprocessor_count << " SMs, "
              << props.total_memory / 1e9 << " GB)\n";
}

// Select specific devices
dm.set_active_devices({0, 2});  // Use GPUs 0 and 2

// Or use all available
dm.set_all_devices();

// Enable P2P access for optimal data transfer
dm.enable_all_peer_access();
```

### Work Distribution Strategies

```cpp
#include <libaccint/device/work_distribution.hpp>

using namespace libaccint::device;

// Round-robin: Simple, deterministic
engine::MultiGPUConfig config;
config.distribution.strategy = DistributionStrategy::RoundRobin;

// Cost-based: Best for load balancing
config.distribution.strategy = DistributionStrategy::CostBased;

// For heterogeneous GPUs, specify relative weights
config.distribution.device_weights = {1.0, 1.5, 1.0, 1.2};  // GPU 1 is 50% faster
```

### Device Context Switching

```cpp
// RAII guard for temporary device switch
{
    ScopedDevice guard(2);  // Switch to GPU 2
    // ... work on GPU 2 ...
}  // Automatically restored to previous device
```

## MPI Distributed Computing

### Prerequisites

Build LibAccInt with MPI support:

```bash
cmake -DLIBACCINT_USE_MPI=ON -DLIBACCINT_USE_CUDA=ON ..
```

### Basic MPI Usage

```cpp
#include <libaccint/mpi/mpi_guard.hpp>
#include <libaccint/mpi/mpi_engine.hpp>

int main(int argc, char** argv) {
    using namespace libaccint::mpi;
    
    // Initialize MPI (safe if already initialized externally)
    MPIGuard mpi(&argc, &argv);
    
    // Create basis set (must be identical on all ranks)
    BasisSet basis(shells);
    
    // Configure MPI engine
    MPIEngineConfig config;
    config.gpu_mapping = GPUMapping::Exclusive;  // One GPU per rank
    
    MPIEngine engine(basis, config);
    
    // Each rank computes its share
    // ... use engine.compute_quartets() ...
    
    // Reduce results
    std::vector<double> local_J = /* ... */;
    std::vector<double> global_J(nbf * nbf);
    engine.allreduce(local_J.data(), global_J.data(), nbf * nbf);
    
    return 0;
}
```

### MPI Threading Contract

LibAccInt alpha supports a funneled hybrid MPI model:

- `MPIGuard` requests `MPI_THREAD_FUNNELED`.
- Intra-rank compute may use local OpenMP or local GPU execution.
- MPI-facing calls such as barriers and reductions must be made from one thread
  per rank.
- Do not mutate device selection or dispatch configuration while a rank is
  actively computing.

### GPU Mapping Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `RoundRobin` | Rank N gets GPU (N % num_gpus) | Multi-rank per node |
| `Packed` | Each rank gets multiple GPUs | Few ranks, many GPUs |
| `Exclusive` | One GPU per rank | Typical distributed setup |
| `UserDefined` | Custom mapping | Complex topologies |

### Running with MPI

```bash
# 4 ranks across 2 nodes, 2 GPUs per node
mpirun -np 4 --hostfile nodes.txt ./my_app

# Using SLURM
srun -N 2 --ntasks-per-node=2 --gpus-per-task=1 ./my_app
```

## Performance Considerations

### Optimal GPU Selection

For systems with multiple GPUs, prefer NVLink-connected pairs:

```cpp
// Automatically select optimal devices
auto optimal_ids = dm.select_optimal_devices(2);  // Best 2 GPUs
config.device_ids = optimal_ids;
```

### Load Balancing

Monitor load balance in production:

```cpp
config.collect_stats = true;
engine.compute_all_eri(fock);

auto& stats = engine.stats();
std::cout << "Load balance efficiency: " 
          << stats.load_balance_efficiency() * 100 << "%\n";
```

### Reduction Overhead

For very large matrices, reduction can become a bottleneck:
- **2 GPUs**: ~2-5% overhead
- **4 GPUs**: ~5-10% overhead
- **8 GPUs**: ~10-15% overhead

Consider overlapping compute and communication for best performance.

## API Reference

### MultiGPUEngine

| Method | Description |
|--------|-------------|
| `device_count()` | Number of active GPUs |
| `engine(i)` | Get CudaEngine for device i |
| `compute_overlap_matrix()` | Compute S matrix |
| `compute_kinetic_matrix()` | Compute T matrix |
| `compute_all_eri(consumer)` | Compute all ERIs with consumer |
| `synchronize_all()` | Synchronize all devices |

### MultiGPUFockBuilder

| Method | Description |
|--------|-------------|
| `set_density(D, nbf)` | Set density matrix (replicated) |
| `reset()` | Reset J/K to zero |
| `get_coulomb_matrix()` | Get reduced J matrix |
| `get_exchange_matrix()` | Get reduced K matrix |
| `get_fock_matrix(H_core, x)` | Get F = H + J - xK |

### DeviceManager

| Method | Description |
|--------|-------------|
| `device_count()` | Total visible GPUs |
| `set_active_devices(ids)` | Select GPUs for use |
| `enable_all_peer_access()` | Enable P2P between active GPUs |
| `can_access_peer(src, dst)` | Check P2P capability |
| `select_optimal_devices(n)` | Select best n GPUs |

## Troubleshooting

### Common Issues

1. **"No CUDA devices available"**
   - Check `nvidia-smi` output
   - Verify `CUDA_VISIBLE_DEVICES` environment variable

2. **Incorrect results with multi-GPU**
   - Ensure all devices compute with same precision
   - Check for race conditions in custom consumers

3. **Poor scaling efficiency**
   - Monitor load balance with `collect_stats`
   - Check for P2P availability
   - Consider larger problem sizes

### Debugging

Enable verbose output:

```cpp
#include <libaccint/device/device_manager.hpp>

auto& dm = DeviceManager::instance();
std::cout << dm.summary() << "\n";
```

Check peer access matrix:

```cpp
for (const auto& info : dm.get_peer_access_matrix()) {
    if (info.source_device != info.target_device) {
        std::cout << info.source_device << " -> " << info.target_device
                  << ": " << (info.can_access ? "P2P" : "HOST") 
                  << (info.nvlink_connected ? " (NVLink)" : "") << "\n";
    }
}
```
