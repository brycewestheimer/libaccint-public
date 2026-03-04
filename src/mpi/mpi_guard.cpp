// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file mpi_guard.cpp
/// @brief MPI guard implementation

#include <libaccint/mpi/mpi_guard.hpp>

#if LIBACCINT_USE_MPI

namespace libaccint::mpi {

// ============================================================================
// MPIGuard Implementation
// ============================================================================

MPIGuard::MPIGuard(int* argc, char*** argv, int required_thread_support)
    : required_thread_level_(required_thread_support) {
    
    int already_initialized = 0;
    MPI_Initialized(&already_initialized);
    
    if (already_initialized) {
        // MPI was initialized externally, don't take ownership
        owns_finalize_ = false;
        
        // Query the actual thread level
        MPI_Query_thread(&provided_thread_level_);
    } else {
        // We're initializing MPI
        owns_finalize_ = true;
        
        int err = MPI_Init_thread(argc, argv, required_thread_support,
                                   &provided_thread_level_);
        if (err != MPI_SUCCESS) {
            throw MPIError("MPI_Init_thread", err);
        }
    }
}

MPIGuard::~MPIGuard() {
    if (owns_finalize_) {
        // Check if already finalized (shouldn't happen, but be safe)
        int finalized = 0;
        MPI_Finalized(&finalized);
        
        if (!finalized) {
            MPI_Finalize();
        }
    }
}

bool MPIGuard::is_initialized() {
    int flag = 0;
    MPI_Initialized(&flag);
    return flag != 0;
}

bool MPIGuard::is_finalized() {
    int flag = 0;
    MPI_Finalized(&flag);
    return flag != 0;
}

int MPIGuard::rank(MPI_Comm comm) {
    int r = 0;
    int err = MPI_Comm_rank(comm, &r);
    if (err != MPI_SUCCESS) {
        throw MPIError("MPI_Comm_rank", err);
    }
    return r;
}

int MPIGuard::size(MPI_Comm comm) {
    int s = 0;
    int err = MPI_Comm_size(comm, &s);
    if (err != MPI_SUCCESS) {
        throw MPIError("MPI_Comm_size", err);
    }
    return s;
}

std::string MPIGuard::processor_name() {
    char name[MPI_MAX_PROCESSOR_NAME];
    int len = 0;
    MPI_Get_processor_name(name, &len);
    return std::string(name, len);
}

void MPIGuard::barrier(MPI_Comm comm) {
    int err = MPI_Barrier(comm);
    if (err != MPI_SUCCESS) {
        throw MPIError("MPI_Barrier", err);
    }
}

}  // namespace libaccint::mpi

#endif  // LIBACCINT_USE_MPI
