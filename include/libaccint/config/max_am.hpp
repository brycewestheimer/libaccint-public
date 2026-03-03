// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file max_am.hpp
/// @brief Configurable maximum angular momentum for generated kernels
///
/// Angular momentum support levels:
///
///   AM  Label  Cartesian components  Status
///   --  -----  --------------------  ------
///    0    S            1              Fully generated and tested
///    1    P            3              Fully generated and tested
///    2    D            6              Fully generated and tested
///    3    F           10              Fully generated and tested
///    4    G           15              Fully generated and tested
///    5    H           21              Deferred (not in stable contract this cycle)
///    6    I           28              Deferred (not in stable contract this cycle)
///
/// CPU generic (runtime-recursive) kernels handle arbitrary AM regardless
/// of this setting.  This constant only controls which AM values have
/// specialized, code-generated kernel implementations.
///
/// Set by CMake via -DLIBACCINT_MAX_AM=<value> (see Step 13.4).
/// The #ifndef guard allows CMake to override with a compile definition.

#ifndef LIBACCINT_MAX_AM
/// Default: G-functions (AM=4).  All kernels up to this level are generated.
#define LIBACCINT_MAX_AM 4
#endif

static_assert(LIBACCINT_MAX_AM >= 0 && LIBACCINT_MAX_AM <= 4,
    "LIBACCINT_MAX_AM must be between 0 (S-functions) and 4 (G-functions) for the stable contract in this cycle");
