# Error Handling Guide

This document describes the exception hierarchy in LibAccInt, how C++ exceptions
map to Python exceptions, GPU error handling, and common error scenarios.

## Exception Hierarchy

LibAccInt uses a structured exception hierarchy rooted in `libaccint::Exception`,
which itself inherits from `std::runtime_error`. All exceptions carry descriptive
messages that include the error category and, where applicable, source file
location information.

```
std::runtime_error
  |
  +-- libaccint::Exception                    (base for all LibAccInt errors)
        |
        +-- InvalidArgumentException          (bad input parameter)
        +-- InvalidStateException             (invalid configuration or state)
        +-- NotImplementedException           (feature not yet available)
        +-- MemoryException                   (resource allocation failure)
        +-- BackendError                      (backend-related error, carries BackendType)
        |     |
        |     +-- memory::CudaError           (CUDA-specific error)
        |
        +-- NumericalException                (overflow, underflow, convergence)
```

### Exception Classes

**`libaccint::Exception`** (defined in `include/libaccint/utils/error_handling.hpp`)

The base exception class. All LibAccInt-specific exceptions inherit from this.

```cpp
class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& message);
};
```

**`InvalidArgumentException`**

Thrown when a function receives an invalid input parameter. The message is
automatically prefixed with `"Invalid argument: "`.

```cpp
throw InvalidArgumentException("basis set must have at least one shell");
```

**`InvalidStateException`**

Thrown when the library is in an invalid state for the requested operation.
The message is prefixed with `"Invalid state: "`.

```cpp
throw InvalidStateException("engine not initialized");
```

**`NotImplementedException`**

Thrown when a feature is not yet implemented. Supports an optional phase
parameter to indicate when the feature is planned.

```cpp
throw NotImplementedException("density fitting");
throw NotImplementedException("analytic Hessians", "Phase 6");
```

**`MemoryException`**

Thrown when memory allocation fails, either on the host or device. The message
is prefixed with `"Memory error: "`.

**`BackendError`** (defined in `include/libaccint/core/backend.hpp`)

Thrown for backend-related errors (availability, initialization, GPU errors).
Inherits from `Exception` and carries the `BackendType` enum value. This is
the base class for `memory::CudaError`.

```cpp
throw BackendError(BackendType::CUDA, "no CUDA-capable device found");
```

**`NumericalException`**

Thrown for numerical issues such as overflow, underflow, or convergence
failure during integral computation. The message is prefixed with
`"Numerical error: "`.

## Assertion Macros

LibAccInt provides convenience macros for common error patterns:

**`LIBACCINT_ASSERT(condition, message)`**

Checks a condition and throws `InvalidStateException` if it evaluates to
false. Includes the source file and line number in the error message.

```cpp
LIBACCINT_ASSERT(buffer.size() >= required,
                 "buffer too small for shell pair");
// Throws: "Invalid state: buffer too small for shell pair (at file.cpp:42)"
```

**`LIBACCINT_NOT_IMPLEMENTED(feature)`**

Throws `NotImplementedException` for an unimplemented feature.

```cpp
LIBACCINT_NOT_IMPLEMENTED("H-type angular momentum");
```

## GPU Error Handling

### CUDA Errors

CUDA API calls are wrapped with the `LIBACCINT_CUDA_CHECK` macro, defined in
`include/libaccint/memory/device_memory.hpp`. On failure, it throws a
`memory::CudaError` exception (which inherits from `BackendError`).

```cpp
// Wraps a CUDA API call -- throws CudaError on failure
LIBACCINT_CUDA_CHECK(cudaMalloc(&ptr, size));
```

The `CudaError` class provides access to the source file and line where the
error was detected:

```cpp
try {
    LIBACCINT_CUDA_CHECK(cudaMalloc(&ptr, very_large_size));
} catch (const memory::CudaError& e) {
    std::cerr << "CUDA error at " << e.file() << ":" << e.line() << "\n";
    std::cerr << e.what() << "\n";
}
```

`CudaError` inherits from `BackendError`, so a catch block for the base
class can handle all GPU errors:

```cpp
try {
    engine.compute_overlap_matrix(S);
} catch (const BackendError& e) {
    std::cerr << "GPU error: " << e.what() << "\n";
}
```

## C++ to Python Exception Mapping

### Automatic Translation via pybind11

LibAccInt's Python bindings (in `python/src/advanced_bindings.cpp`) register
`BackendError` as a Python exception that inherits from `RuntimeError`:

```cpp
py::register_exception<BackendError>(m, "BackendError", PyExc_RuntimeError);
```

In Python, you can catch it as:

```python
import libaccint

try:
    engine = libaccint.Engine(basis)
    S = engine.compute_overlap_matrix()
except libaccint.BackendError as e:
    print(f"Backend error: {e}")
except RuntimeError as e:
    # Catches other C++ exceptions that inherit from std::runtime_error,
    # including InvalidArgumentException, InvalidStateException, etc.
    print(f"LibAccInt error: {e}")
```

### Exception Mapping Table

| C++ Exception               | Python Exception             |
|-----------------------------|------------------------------|
| `BackendError`              | `libaccint.BackendError`     |
| `InvalidArgumentException`  | `RuntimeError`               |
| `InvalidStateException`     | `RuntimeError`               |
| `NotImplementedException`   | `RuntimeError`               |
| `MemoryException`           | `RuntimeError`               |
| `NumericalException`        | `RuntimeError`               |
| `std::invalid_argument`     | `ValueError`                 |
| `std::runtime_error`        | `RuntimeError`               |
| `std::exception`            | `RuntimeError`               |

The pybind11 framework automatically translates standard C++ exceptions. Since
most LibAccInt exceptions inherit from `std::runtime_error`, they appear as
`RuntimeError` in Python. The explicit registration of `BackendError` provides
a more specific exception type for backend availability issues.

## Common Error Scenarios

### 1. No GPU Available

**Symptom:** `BackendError` when requesting GPU execution.

```python
try:
    S = engine.compute_overlap_matrix(hint=libaccint.BackendHint.ForceGPU)
except libaccint.BackendError:
    # Fall back to CPU
    S = engine.compute_overlap_matrix(hint=libaccint.BackendHint.ForceCPU)
```

**Recommendation:** Use `BackendHint.Auto` (the default) to let the dispatch
policy handle fallback automatically.

### 2. Invalid Basis Set Name

**Symptom:** Exception when creating a basis set with an unrecognized name.

```python
try:
    basis = libaccint.basis_set("unknown-basis", atoms)
except RuntimeError as e:
    print(f"Basis set error: {e}")
```

### 3. Mismatched Matrix Dimensions

**Symptom:** Exception when passing a density matrix with wrong dimensions.

```python
try:
    F = libaccint.build_fock(engine, wrong_size_density, H)
except ValueError as e:
    print(f"Dimension mismatch: {e}")
```

### 4. GPU Out of Memory

**Symptom:** `CudaError` during large computations.

```cpp
try {
    engine.compute_eri_device_scatter(result);
} catch (const memory::CudaError& e) {
    // Likely out of GPU memory for large basis sets
    // Fall back to CPU or use a pipelined approach
    cpu_engine.compute_and_consume(op, consumer);
}
```

**Recommendation:** For large basis sets, prefer the primary
`Engine::compute_and_consume(...)` API. The lower-level pipelined/device-scatter
CUDA entry points are experimental specialist APIs and may change during the
alpha cycle.

### 5. Wrong Operator Type

**Symptom:** `InvalidArgumentException` when passing a one-electron operator
where a two-electron operator is expected, or vice versa.

```cpp
// This will throw -- overlap is a one-electron operator
engine.compute_and_consume(Operator::overlap(), fock_builder);
// Error: "compute_and_consume requires a two-electron operator, got: Overlap"
```

### 6. OpenMP Not Available

**Symptom:** Parallel functions silently fall back to single-threaded execution.

```python
import libaccint
print(f"OpenMP available: {libaccint.has_openmp()}")
```

When OpenMP is not compiled in, `ThreadConfig::openmp_available()` returns
`false` and parallel methods execute sequentially. This is not an error, but
performance will be reduced.
