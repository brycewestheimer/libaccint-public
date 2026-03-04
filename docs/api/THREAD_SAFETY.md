# Thread Safety and Parallelism

## Version: 0.1.0-alpha.2

This is the authoritative support matrix for LibAccInt alpha concurrency.

Legend:

| Term | Meaning |
|------|---------|
| Supported | Part of the alpha contract when the listed conditions are met |
| Unsupported | Not part of the alpha contract; external synchronization or a different API is required |
| Funneled MPI | MPI calls must be made from one thread per rank |

## Alpha Contract

- MPI support is funneled hybrid.
- CPU shared-engine compute is narrow: use one `Engine` per thread or the explicit `_parallel` APIs.
- GPU shared-engine support is limited to batched APIs that use per-call execution slots.
- Configuration and state mutation must happen before compute begins.
- "Supported" means correctness-supported. It does not imply optimal performance under all threading setups.

## Support Matrix

| Object / API | Concurrent shared-instance use | Supported conditions | Unsupported conditions |
|--------------|-------------------------------|----------------------|------------------------|
| `BasisSet` const access | Supported | Concurrent reads, including first `shell_set_pairs()` / `shell_set_quartets()` cache generation | Concurrent `clear_work_unit_cache()` while any thread still holds prior references |
| `Shell`, `ShellSet`, `ShellSetPair`, `ShellSetQuartet`, `Operator` | Supported | Read-only use after construction | Concurrent mutation |
| `Engine` CPU convenience methods (`compute_overlap_matrix`, `compute_kinetic_matrix`, `compute_nuclear_matrix`, `compute_core_hamiltonian`, serial `compute*`) | Unsupported | One engine per thread | Concurrent calls on the same CPU-backed `Engine` |
| `Engine::compute_and_consume_parallel` | Supported | CPU/OpenMP only; consumer must satisfy `ParallelConsumer`; do not call re-entrantly on the same `Engine` | Assuming this dispatches to GPU; concurrent re-entrant use on one `Engine` |
| `Engine::compute_1e_parallel` | Supported | CPU/OpenMP only; one call at a time per `Engine` | Re-entrant use on one `Engine` |
| `Engine::compute_batch_parallel` / `compute_all_2e_parallel` | Supported | Multiple host threads may share a GPU-backed engine; on CPU this remains library-managed OpenMP parallelism | Concurrent config mutation during compute |
| `Engine::compute_batch` | Supported on GPU | Shared GPU-backed `Engine`, read-only basis, no concurrent mutators | Shared CPU-backed `Engine` serial path |
| `Engine::compute_shell_set_pair` | Supported on GPU | Shared GPU-backed `Engine`, read-only state | Concurrent config mutation, device selection changes during compute |
| `Engine::compute_shell_set_quartet` | Supported conditionally | GPU path with corrected stream ordering for `GpuFockBuilder`; CPU path only through library-managed parallel consumer flow | Treating all consumers or all backends as generally shared-thread-safe |
| `CudaEngine::compute_eri_batch_device_handle` | Supported | Device pointer used only while handle remains alive | Retaining raw device pointer after handle destruction |
| `CudaEngine::compute_eri_batch_device(quartet, double*&, size_t&)` | Unsupported | None; kept only as a deprecated compatibility shim | Any use in alpha; it fails fast at runtime |
| `CudaEngine::set_dispatch_config` | Unsupported during compute | Call before any concurrent GPU work begins | Mutating slot count or dispatch thresholds while compute is active |
| `DispatchConfig::n_gpu_slots` | Supported | Positive values only (`>= 1`) | `0` |
| `EriPipelineConfig::n_slots` | Supported | Positive values only (`>= 1`) | `0` |
| `DeviceManager` selection / active-device mutation | Unsupported during compute | Configure devices at startup | Concurrent reads/writes while work is running |
| `ThreadConfig::set_num_threads` / `reset` | Process-global only | Set before compute phases | Treating it as per-engine or thread-local |
| `FockBuilder` with `Sequential` | Unsupported | One builder per thread or external synchronization | Shared concurrent accumulation |
| `FockBuilder` with `Atomic` | Supported | Shared CPU accumulation | Assuming no overhead tradeoff |
| `FockBuilder` with `ThreadLocal` | Supported | `prepare_parallel()` / `finalize_parallel()` around the parallel region | Omitting lifecycle bookends |
| `GpuFockBuilder` | Supported for corrected single-flow GPU accumulation | Configure/reset/read results outside active compute; use corrected producer/consumer ordering | Treating one builder as a generally shared multithreaded object |
| `MPIGuard` | Supported | Requests `MPI_THREAD_FUNNELED`; safe if MPI was already initialized externally | Assuming `MPI_THREAD_SERIALIZED` or `MPI_THREAD_MULTIPLE` |
| `MPIEngine` | Supported | Rank-partitioned CPU or GPU compute; MPI collectives remain funneled | Concurrent MPI collectives from multiple threads on one rank; mutating local engine/device config during compute |
| `MPIFockBuilder` local CPU accumulation | Supported | Can use `set_threading_strategy()`, `prepare_parallel()`, `finalize_parallel()` for per-rank CPU parallel accumulation | Assuming reductions are thread-safe collectives |
| `MPIFockBuilder` local GPU-backed accumulation | Supported | GPU/multi-GPU engine transfers ERIs to host, then accumulates locally before MPI reduction | Treating it as a device-side MPI consumer in alpha |
| `MPIFockBuilder::allreduce()` / `reduce_to_root()` | Funneled MPI only | One calling thread per rank after local accumulation is complete | Concurrent collectives on the same communicator from multiple threads |
| `MemoryPool` / `GpuSlotPool` | Supported | Internal mutex/condition variable protection | N/A |
| `BatchBufferPool` / `BufferPool` | Unsupported | One instance per thread | Shared concurrent use |

## Recommended Patterns

### CPU: one engine per thread

```cpp
#pragma omp parallel
{
    libaccint::Engine thread_engine(basis);
    std::vector<libaccint::Real> S;
    thread_engine.compute_overlap_matrix(S);
}
```

### CPU: explicit `_parallel` API

```cpp
libaccint::Engine engine(basis);
libaccint::consumers::FockBuilder fock(nbf);
fock.set_threading_strategy(
    libaccint::consumers::FockThreadingStrategy::ThreadLocal);
fock.set_density(D.data(), nbf);
fock.prepare_parallel(threads);
engine.compute_and_consume_parallel(libaccint::Operator::coulomb(), fock, threads);
fock.finalize_parallel();
```

### GPU: shared engine with batched APIs

```cpp
libaccint::DispatchConfig config;
config.n_gpu_slots = 2;
libaccint::Engine engine(basis, config);

#pragma omp parallel for schedule(dynamic)
for (libaccint::Size i = 0; i < quartets.size(); ++i) {
    results[i] = engine.compute_batch(
        libaccint::Operator::coulomb(), quartets[i], libaccint::BackendHint::ForceGPU);
}
```

### MPI: funneled hybrid usage

```cpp
libaccint::mpi::MPIGuard mpi(&argc, &argv);  // requests MPI_THREAD_FUNNELED

libaccint::mpi::MPIEngine engine(basis, config);
libaccint::mpi::MPIFockBuilder fock(engine.comm(), basis.n_basis_functions());
fock.set_density(D.data(), basis.n_basis_functions());

// Local per-rank compute may use CPU OpenMP or local GPU execution.
engine.compute_all_eri(fock);

// Collective remains funneled: one calling thread per rank.
fock.allreduce();
```

## Important Caveats

- `clear_work_unit_cache()` invalidates previously returned pair/quartet references.
- `compute_and_consume_screened_parallel()` is CPU/OpenMP work; set up any shared consumer state before entering the call.
- `set_density_matrix()` for density-weighted screening is not a concurrent mutation API.
- Full-matrix convenience methods are documented separately from batched shared-GPU concurrency. Do not assume every GPU-dispatched method is safe to call concurrently on one shared object.
