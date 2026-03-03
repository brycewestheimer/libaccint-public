// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file mpi_utils.hpp
/// @brief MPI utility helpers — safe integer conversions, chunked operations
///
/// MPI functions accept `int` count parameters. When the codebase uses
/// `size_t` (aliased as `Size`) for buffer lengths, a direct cast can
/// silently overflow for large basis sets. This header provides
/// `safe_mpi_count()` for validation and `chunked_reduce()` /
/// `chunked_allreduce()` for data exceeding INT_MAX elements.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_MPI

#include <climits>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include <mpi.h>

namespace libaccint::mpi {

/// @brief Safely convert a size_t count to int for MPI functions
///
/// MPI uses `int` for element counts. This function checks that the
/// value fits and throws a descriptive `std::overflow_error` if it
/// exceeds `INT_MAX`.
///
/// @param count Element count as size_t
/// @return The same value as int
/// @throws std::overflow_error if count > INT_MAX
inline int safe_mpi_count(std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "MPI count exceeds INT_MAX (" +
            std::to_string(std::numeric_limits<int>::max()) +
            "): " + std::to_string(count) +
            ". Consider splitting the operation into smaller chunks.");
    }
    return static_cast<int>(count);
}

/// @brief Maximum chunk size for chunked MPI operations
inline constexpr std::size_t MPI_CHUNK_SIZE = 
    static_cast<std::size_t>(std::numeric_limits<int>::max());

/// @brief Chunked MPI_Reduce that handles count > INT_MAX
///
/// Splits the reduction into multiple MPI_Reduce calls, each with at
/// most INT_MAX elements, so arbitrarily large buffers can be reduced.
///
/// @param local_data  Source buffer (on all ranks)
/// @param global_data Destination buffer (valid only on root)
/// @param count       Total number of elements
/// @param root        Root rank
/// @param comm        MPI communicator
/// @throws MPIError on MPI failure
inline void chunked_reduce(const double* local_data, double* global_data,
                           std::size_t count, int root, MPI_Comm comm) {
    std::size_t offset = 0;
    while (offset < count) {
        std::size_t remaining = count - offset;
        int chunk = static_cast<int>(std::min(remaining, MPI_CHUNK_SIZE));
        int err = MPI_Reduce(local_data + offset, global_data + offset,
                              chunk, MPI_DOUBLE, MPI_SUM, root, comm);
        if (err != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Reduce failed in chunk at offset " +
                                      std::to_string(offset));
        }
        offset += static_cast<std::size_t>(chunk);
    }
}

/// @brief Chunked MPI_Allreduce that handles count > INT_MAX
///
/// Splits the all-reduction into multiple MPI_Allreduce calls, each with at
/// most INT_MAX elements.
///
/// @param local_data  Source buffer
/// @param global_data Destination buffer (valid on all ranks)
/// @param count       Total number of elements
/// @param comm        MPI communicator
/// @throws MPIError on MPI failure
inline void chunked_allreduce(const double* local_data, double* global_data,
                              std::size_t count, MPI_Comm comm) {
    std::size_t offset = 0;
    while (offset < count) {
        std::size_t remaining = count - offset;
        int chunk = static_cast<int>(std::min(remaining, MPI_CHUNK_SIZE));
        int err = MPI_Allreduce(local_data + offset, global_data + offset,
                                 chunk, MPI_DOUBLE, MPI_SUM, comm);
        if (err != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Allreduce failed in chunk at offset " +
                                      std::to_string(offset));
        }
        offset += static_cast<std::size_t>(chunk);
    }
}

/// @brief Chunked in-place MPI_Allreduce that handles count > INT_MAX
///
/// @param data  Buffer for in-place reduction (both input and output)
/// @param count Total number of elements
/// @param comm  MPI communicator
/// @throws MPIError on MPI failure
inline void chunked_allreduce_inplace(double* data, std::size_t count,
                                       MPI_Comm comm) {
    std::size_t offset = 0;
    while (offset < count) {
        std::size_t remaining = count - offset;
        int chunk = static_cast<int>(std::min(remaining, MPI_CHUNK_SIZE));
        int err = MPI_Allreduce(MPI_IN_PLACE, data + offset,
                                 chunk, MPI_DOUBLE, MPI_SUM, comm);
        if (err != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Allreduce (in-place) failed in chunk at offset " +
                                      std::to_string(offset));
        }
        offset += static_cast<std::size_t>(chunk);
    }
}

/// @brief Chunked in-place MPI_Reduce that handles count > INT_MAX
///
/// @param data  Buffer for in-place reduction (both input and output on root)
/// @param count Total number of elements
/// @param root  Root rank
/// @param rank  This rank's ID
/// @param comm  MPI communicator
/// @throws MPIError on MPI failure
inline void chunked_reduce_inplace(double* data, std::size_t count,
                                    int root, int rank, MPI_Comm comm) {
    std::size_t offset = 0;
    while (offset < count) {
        std::size_t remaining = count - offset;
        int chunk = static_cast<int>(std::min(remaining, MPI_CHUNK_SIZE));
        if (rank == root) {
            int err = MPI_Reduce(MPI_IN_PLACE, data + offset,
                                  chunk, MPI_DOUBLE, MPI_SUM, root, comm);
            if (err != MPI_SUCCESS) {
                throw std::runtime_error("MPI_Reduce (in-place) failed in chunk at offset " +
                                          std::to_string(offset));
            }
        } else {
            int err = MPI_Reduce(data + offset, nullptr,
                                  chunk, MPI_DOUBLE, MPI_SUM, root, comm);
            if (err != MPI_SUCCESS) {
                throw std::runtime_error("MPI_Reduce failed in chunk at offset " +
                                          std::to_string(offset));
            }
        }
        offset += static_cast<std::size_t>(chunk);
    }
}

}  // namespace libaccint::mpi

#else  // !LIBACCINT_USE_MPI

#include <climits>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace libaccint::mpi {

/// @brief safe_mpi_count stub for non-MPI builds (used in tests)
inline int safe_mpi_count(std::size_t count) {
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "MPI count exceeds INT_MAX: " + std::to_string(count));
    }
    return static_cast<int>(count);
}

}  // namespace libaccint::mpi

#endif  // LIBACCINT_USE_MPI
