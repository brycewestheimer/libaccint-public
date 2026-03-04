// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_device_operator_data.cpp
/// @brief DISABLED tests for device (GPU) operator data upload/free
///
/// These tests are disabled because they require a CUDA device.
/// They document what should be tested when running on GPU builds.

#include <libaccint/operators/operator_types.hpp>
#include <gtest/gtest.h>

namespace libaccint::testing {

// ============================================================================
// DeviceOperatorData tests — disabled for CPU-only builds
// ============================================================================

/// @brief Test uploading point charge data to device memory
/// Should verify:
/// - DevicePointChargeData is correctly allocated
/// - Host→device copy preserves positions and charges
/// - valid() returns true after upload
TEST(DeviceOperatorDataTest, DISABLED_UploadPointCharges) {
    // In a GPU build, this would:
    // 1. Create PointChargeParams with test data (e.g., water molecule)
    // 2. Upload via PointChargeDeviceHandle
    // 3. Verify handle.data().valid() == true
    // 4. Verify handle.data().n_charges matches input
    SUCCEED();
}

/// @brief Test freeing device point charge data
/// Should verify:
/// - After PointChargeDeviceHandle destruction, device memory is freed
/// - Double-free does not occur (RAII)
/// - Default-constructed handle has valid() == false
TEST(DeviceOperatorDataTest, DISABLED_FreePointChargeData) {
    // In a GPU build, this would:
    // 1. Create a PointChargeDeviceHandle in a scope
    // 2. Let it go out of scope
    // 3. Verify no CUDA errors after destruction
    SUCCEED();
}

/// @brief Test uploading distributed multipole data to device
/// Should verify:
/// - Charge, dipole, and quadrupole components uploaded correctly
/// - max_rank metadata preserved
TEST(DeviceOperatorDataTest, DISABLED_UploadDistributedMultipole) {
    SUCCEED();
}

/// @brief Test device data with empty parameters
/// Should verify graceful handling of zero-length arrays
TEST(DeviceOperatorDataTest, DISABLED_EmptyParams) {
    SUCCEED();
}

}  // namespace libaccint::testing
