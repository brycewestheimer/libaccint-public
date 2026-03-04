# API Stability Guarantees and Deprecation Policy

## Version: 0.1.0-alpha.2
## Status: Alpha (pre-release)

> **Note:** LibAccInt is currently in alpha. APIs listed below as Tier 1 (Stable)
> are expected to remain stable through the 0.x series, but breaking changes may
> occur before 1.0.0. Tier 2 (Provisional) APIs are more likely to change.

---

## Stability Tiers

LibAccInt uses a three-tier stability classification for its public API:

### Tier 1: Stable API (guaranteed backward compatible)

These APIs will not change in backward-incompatible ways within a major version. Any changes will go through the deprecation process described below.

| Component | Header | Since |
|-----------|--------|-------|
| `Shell` | `basis/shell.hpp` | 0.1.0 |
| `BasisSet` | `basis/basis_set.hpp` | 0.1.0 |
| `Engine` | `engine/engine.hpp` | 0.1.0 |
| `Operator` | `operators/operator.hpp` | 0.1.0 |
| `OperatorKind` | `operators/operator_types.hpp` | 0.1.0 |
| `OneElectronOperator` | `operators/one_electron_operator.hpp` | 0.1.0 |
| `FockBuilder` | `consumers/fock_builder.hpp` | 0.1.0 |
| `Point3D`, `Real`, `Index`, `Size` | `core/types.hpp` | 0.1.0 |
| `PointChargeParams` | `operators/operator_types.hpp` | 0.1.0 |
| `create_builtin_basis()` | `data/builtin_basis.hpp` | 0.1.0 |
| `load_basis_set()` | `data/basis_parser.hpp` | 0.1.0 |
| `version()` | `config.hpp` | 0.1.0 |

### Tier 2: Provisional API (may change with notice)

These APIs are functional and tested but may undergo refinement in minor versions.

| Component | Header | Since |
|-----------|--------|-------|
| `DFFockBuilder` | `consumers/df_fock_builder.hpp` | 0.1.0 |
| `AuxiliaryBasisSet` | `basis/auxiliary_basis_set.hpp` | 0.1.0 |
| `BseJsonParser` | `data/bse_json_parser.hpp` | 0.1.0 |
| `ScreeningOptions` | `screening/screening_options.hpp` | 0.1.0 |
| `DispatchConfig` | `engine/dispatch_policy.hpp` | 0.1.0 |
| `MixedPrecisionFockBuilder` | `consumers/mixed_precision_fock_builder.hpp` | 0.1.0 |

### Tier 3: Internal/Experimental API (no stability guarantee)

These components are not part of the public API and may change without notice.

- Everything in `libaccint::detail::` namespace
- Headers in `engine/` other than `engine.hpp`
- Device-specific headers (`device/`, `cuda/`)
- Math internals (`math/` --- use through Engine)
- Kernel registry internals
- `MultiGPUEngine`
- `MultiGPUFockBuilder`
- `CudaEngine::compute_eri_pipelined()`
- `CudaEngine::compute_eri_device_scatter()`

---

## Semantic Versioning

LibAccInt follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (e.g., 1.x.x -> 2.0.0): Breaking API changes
- **MINOR** (e.g., 1.0.x -> 1.1.0): New features, backward compatible
- **PATCH** (e.g., 1.0.0 -> 1.0.1): Bug fixes, no API changes

### ABI Compatibility

- **During alpha (0.x)**: ABI stability is NOT guaranteed. Recompilation may be required between any releases.
- **Within a patch version** (post-1.0): ABI is guaranteed stable
- **Within a minor version** (post-1.0): ABI stability is best-effort
- **Across major versions**: No ABI guarantees

---

## Deprecation Policy

### Process

1. **Announcement**: Deprecated APIs are marked with `[[deprecated("message")]]` in the header and documented in release notes
2. **Grace Period**: Deprecated APIs remain functional for at least **2 minor releases** (e.g., deprecated in 1.2, removed earliest in 1.4)
3. **Removal**: Removed in the next major version (or after the grace period in a minor version if announced)

### How to Check for Deprecations

```cpp
// Deprecated functions emit compiler warnings:
// warning: 'old_function' is deprecated: Use new_function instead [-Wdeprecated-declarations]

// Example deprecation pattern:
[[deprecated("Use Engine::compute() instead")]]
void Engine::compute_legacy(/* ... */);
```

### Migration Support

When an API is deprecated:
- The deprecation message includes the replacement API
- The migration guide is updated with examples
- Both old and new APIs work during the grace period

---

## Header Inclusion Guarantees

Including `<libaccint/libaccint.hpp>` provides all Tier 1 stable APIs. Individual headers can be included for finer-grained control:

```cpp
// Full API (recommended for most users)
#include <libaccint/libaccint.hpp>

// Minimal includes for specific functionality
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/basis_set.hpp>
#include <libaccint/engine/engine.hpp>
```

---

## Platform Support

| Platform | Compiler | Status |
|----------|----------|--------|
| Linux x86_64 | GCC 12+ | Full support |
| Linux x86_64 | Clang 16+ | Full support |
| Linux x86_64 + NVIDIA GPU | GCC 12+ / nvcc 12+ | Full support |
| macOS ARM64 | Apple Clang 15+ | CPU only |
| Windows x64 | MSVC 2022+ | Experimental |

---

## Reporting API Issues

If you find an API inconsistency or have a suggestion:

1. Open an issue with the `api` label
2. Tag with the affected version
3. Include a minimal reproducing example
