// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file libaccint.hpp
/// @brief Umbrella header for LibAccInt v0.1.0 (MVP)
///
/// Including this header provides access to all public LibAccInt types
/// and functions. For finer-grained control, include individual headers.

// Core types and configuration
#include <libaccint/core/types.hpp>
#include <libaccint/core/backend.hpp>
#include <libaccint/core/precision.hpp>
#include <libaccint/utils/constants.hpp>
#include <libaccint/utils/error_handling.hpp>

// Basis set
#include <libaccint/basis/shell.hpp>
#include <libaccint/basis/shell_set.hpp>
#include <libaccint/basis/shell_set_pair.hpp>
#include <libaccint/basis/shell_set_quartet.hpp>
#include <libaccint/basis/basis_set.hpp>

// Operators
#include <libaccint/operators/operator_types.hpp>
#include <libaccint/operators/operator.hpp>
#include <libaccint/operators/one_electron_operator.hpp>

// Buffers
#include <libaccint/buffers/one_electron_buffer.hpp>
#include <libaccint/buffers/two_electron_buffer.hpp>

// Engine
#include <libaccint/engine/engine.hpp>

// Consumers
#include <libaccint/consumers/fock_builder.hpp>

// Data (built-in basis sets, parser)
#include <libaccint/data/builtin_basis.hpp>
#include <libaccint/data/basis_parser.hpp>

// Math utilities
#include <libaccint/math/gaussian_product.hpp>
#include <libaccint/math/boys_function.hpp>
#include <libaccint/math/rys_quadrature.hpp>
#include <libaccint/math/normalization.hpp>
#include <libaccint/math/cartesian_indices.hpp>

// Screening
#include <libaccint/screening/screening_options.hpp>
#include <libaccint/screening/schwarz_bounds.hpp>
#include <libaccint/screening/density_screening.hpp>
#include <libaccint/screening/screened_quartet_iterator.hpp>

// Utility functions
#include <libaccint/utils/matrix_assembly.hpp>

// GPU headers (CUDA backend)
#if LIBACCINT_USE_CUDA
#include <libaccint/engine/cuda_engine.hpp>
#include <libaccint/engine/multi_gpu_engine.hpp>
#include <libaccint/consumers/fock_builder_gpu.hpp>
#include <libaccint/consumers/multi_gpu_fock_builder.hpp>
#endif  // LIBACCINT_USE_CUDA

// MPI headers
#if LIBACCINT_USE_MPI
#include <libaccint/mpi/mpi_guard.hpp>
#endif  // LIBACCINT_USE_MPI

// Density fitting (DF) headers
#if LIBACCINT_USE_DF
#include <libaccint/df/three_center_storage.hpp>
#include <libaccint/df/b_tensor_storage.hpp>
#include <libaccint/df/out_of_core_storage.hpp>
#include <libaccint/data/auxiliary_basis_parser.hpp>
#include <libaccint/data/auxiliary_basis_data.hpp>
#include <libaccint/basis/auxiliary_basis_set.hpp>
#include <libaccint/consumers/df_fock_builder.hpp>
#endif  // LIBACCINT_USE_DF
