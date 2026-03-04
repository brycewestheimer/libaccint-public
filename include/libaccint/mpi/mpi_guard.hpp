// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file mpi_guard.hpp
/// @brief RAII-based MPI initialization and finalization
///
/// Provides safe MPI lifecycle management that integrates with LibAccInt
/// without conflicting with external MPI initialization.

#include <libaccint/config.hpp>

#if LIBACCINT_USE_MPI

#include <mpi.h>
#include <string>
#include <stdexcept>

namespace libaccint::mpi {

/// @brief Exception for MPI-related errors
class MPIError : public std::runtime_error {
public:
    explicit MPIError(const std::string& message)
        : std::runtime_error("MPIError: " + message) {}
    
    MPIError(const std::string& function, int error_code)
        : std::runtime_error("MPIError in " + function + ": " + 
                             error_string(error_code)),
          error_code_(error_code) {}
    
    [[nodiscard]] int error_code() const noexcept { return error_code_; }
    
private:
    static std::string error_string(int error_code) {
        char buffer[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(error_code, buffer, &len);
        return std::string(buffer, len);
    }
    
    int error_code_ = MPI_SUCCESS;
};

/// @brief RAII guard for MPI initialization and finalization
///
/// MPIGuard handles MPI lifecycle in a way that's compatible with both:
/// 1. LibAccInt initializing MPI itself
/// 2. External code (e.g., host application) initializing MPI before LibAccInt
///
/// If MPI is already initialized when MPIGuard is constructed, it will
/// NOT call MPI_Finalize on destruction (to avoid double-finalization).
///
/// Usage:
/// @code
///   // Case 1: LibAccInt manages MPI
///   {
///       MPIGuard mpi;  // Calls MPI_Init_thread
///       // ... do MPI work ...
///   }  // Calls MPI_Finalize
///   
///   // Case 2: External initialization
///   MPI_Init(&argc, &argv);  // External code
///   {
///       MPIGuard mpi;  // Detects existing init, won't finalize
///       // ... do MPI work ...
///   }  // Does NOT call MPI_Finalize
///   MPI_Finalize();  // External code
/// @endcode
class MPIGuard {
public:
    /// @brief Initialize MPI if not already initialized
    /// @param argc Pointer to argument count (may be nullptr)
    /// @param argv Pointer to argument vector (may be nullptr)
    /// @param required_thread_support Thread level (default: MPI_THREAD_FUNNELED)
    explicit MPIGuard(int* argc = nullptr, char*** argv = nullptr,
                      int required_thread_support = MPI_THREAD_FUNNELED);
    
    /// @brief Finalize MPI if we initialized it
    ~MPIGuard();
    
    // Non-copyable, non-movable
    MPIGuard(const MPIGuard&) = delete;
    MPIGuard& operator=(const MPIGuard&) = delete;
    MPIGuard(MPIGuard&&) = delete;
    MPIGuard& operator=(MPIGuard&&) = delete;
    
    /// @brief Check if we own the MPI finalization
    [[nodiscard]] bool owns_finalize() const noexcept { return owns_finalize_; }
    
    /// @brief Get the provided thread support level
    [[nodiscard]] int thread_level() const noexcept { return provided_thread_level_; }
    
    /// @brief Check if the requested thread support was provided
    [[nodiscard]] bool thread_support_satisfied() const noexcept {
        return provided_thread_level_ >= required_thread_level_;
    }
    
    // =========================================================================
    // Static Query Functions
    // =========================================================================
    
    /// @brief Check if MPI is currently initialized
    [[nodiscard]] static bool is_initialized();
    
    /// @brief Check if MPI has been finalized
    [[nodiscard]] static bool is_finalized();
    
    /// @brief Get the rank of the current process
    /// @param comm Communicator (default: MPI_COMM_WORLD)
    [[nodiscard]] static int rank(MPI_Comm comm = MPI_COMM_WORLD);
    
    /// @brief Get the size of the communicator
    /// @param comm Communicator (default: MPI_COMM_WORLD)
    [[nodiscard]] static int size(MPI_Comm comm = MPI_COMM_WORLD);
    
    /// @brief Get the processor name
    [[nodiscard]] static std::string processor_name();
    
    /// @brief Barrier synchronization
    /// @param comm Communicator (default: MPI_COMM_WORLD)
    static void barrier(MPI_Comm comm = MPI_COMM_WORLD);
    
    /// @brief Check if this is the root process (rank 0)
    [[nodiscard]] static bool is_root(MPI_Comm comm = MPI_COMM_WORLD) {
        return rank(comm) == 0;
    }

private:
    bool owns_finalize_ = false;
    int required_thread_level_ = MPI_THREAD_FUNNELED;
    int provided_thread_level_ = MPI_THREAD_SINGLE;
};

}  // namespace libaccint::mpi

#else  // !LIBACCINT_USE_MPI

// Provide stub types when MPI is not enabled
namespace libaccint::mpi {

class MPIGuard {
public:
    explicit MPIGuard(int* = nullptr, char*** = nullptr, int = 0) {}
    [[nodiscard]] bool owns_finalize() const noexcept { return false; }
    [[nodiscard]] int thread_level() const noexcept { return 0; }
    [[nodiscard]] bool thread_support_satisfied() const noexcept { return true; }
    
    [[nodiscard]] static bool is_initialized() { return false; }
    [[nodiscard]] static bool is_finalized() { return true; }
    [[nodiscard]] static int rank(void* = nullptr) { return 0; }
    [[nodiscard]] static int size(void* = nullptr) { return 1; }
    [[nodiscard]] static std::string processor_name() { return "localhost"; }
    static void barrier(void* = nullptr) {}
    [[nodiscard]] static bool is_root(void* = nullptr) { return true; }
};

}  // namespace libaccint::mpi

#endif  // LIBACCINT_USE_MPI
