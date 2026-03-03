// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_two_electron_buffer.cpp
/// @brief Unit tests for TwoElectronBuffer

#include <gtest/gtest.h>
#include <libaccint/buffers/two_electron_buffer.hpp>
#include <libaccint/core/types.hpp>

namespace libaccint {

// ============================================================================
// Basic Construction and Properties (DerivOrder=0)
// ============================================================================

TEST(TwoElectronBufferTest, DefaultConstructorCreatesEmptyBuffer) {
    TwoElectronBuffer<0> buffer;

    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.n_integrals(), 0);
    EXPECT_EQ(buffer.na(), 0);
    EXPECT_EQ(buffer.nb(), 0);
    EXPECT_EQ(buffer.nc(), 0);
    EXPECT_EQ(buffer.nd(), 0);
}

TEST(TwoElectronBufferTest, ConstructorWithDimensionsAllocatesCorrectly) {
    TwoElectronBuffer<0> buffer(3, 3, 3, 3);

    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.na(), 3);
    EXPECT_EQ(buffer.nb(), 3);
    EXPECT_EQ(buffer.nc(), 3);
    EXPECT_EQ(buffer.nd(), 3);
    EXPECT_EQ(buffer.n_integrals(), 81);  // 3*3*3*3
    EXPECT_EQ(buffer.size(), 81);         // N_DERIV=1 for DerivOrder=0
}

TEST(TwoElectronBufferTest, ResizeChangeDimensions) {
    TwoElectronBuffer<0> buffer;

    buffer.resize(2, 3, 4, 5);

    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.na(), 2);
    EXPECT_EQ(buffer.nb(), 3);
    EXPECT_EQ(buffer.nc(), 4);
    EXPECT_EQ(buffer.nd(), 5);
    EXPECT_EQ(buffer.n_integrals(), 120);  // 2*3*4*5
    EXPECT_EQ(buffer.size(), 120);
}

TEST(TwoElectronBufferTest, ClearZerosAllElements) {
    TwoElectronBuffer<0> buffer(2, 2, 2, 2);

    // Fill with non-zero values
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < 2; ++c) {
                for (int d = 0; d < 2; ++d) {
                    buffer(a, b, c, d) = 1.0 + a + b + c + d;
                }
            }
        }
    }

    // Verify non-zero
    EXPECT_NE(buffer(1, 1, 1, 1), 0.0);

    // Clear and verify all zeros
    buffer.clear();

    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < 2; ++c) {
                for (int d = 0; d < 2; ++d) {
                    EXPECT_EQ(buffer(a, b, c, d), 0.0)
                        << "Non-zero at (" << a << "," << b << "," << c << "," << d << ")";
                }
            }
        }
    }
}

// ============================================================================
// Accessor Tests (DerivOrder=0)
// ============================================================================

TEST(TwoElectronBufferTest, AccessorCorrectIndexing) {
    TwoElectronBuffer<0> buffer(2, 2, 2, 2);
    buffer.clear();

    // Set specific values
    buffer(0, 0, 0, 0) = 1.0;
    buffer(1, 0, 0, 0) = 2.0;
    buffer(0, 1, 0, 0) = 3.0;
    buffer(0, 0, 1, 0) = 4.0;
    buffer(0, 0, 0, 1) = 5.0;
    buffer(1, 1, 1, 1) = 6.0;

    // Verify retrieval
    EXPECT_EQ(buffer(0, 0, 0, 0), 1.0);
    EXPECT_EQ(buffer(1, 0, 0, 0), 2.0);
    EXPECT_EQ(buffer(0, 1, 0, 0), 3.0);
    EXPECT_EQ(buffer(0, 0, 1, 0), 4.0);
    EXPECT_EQ(buffer(0, 0, 0, 1), 5.0);
    EXPECT_EQ(buffer(1, 1, 1, 1), 6.0);

    // Verify other elements remain zero
    EXPECT_EQ(buffer(0, 0, 1, 1), 0.0);
    EXPECT_EQ(buffer(1, 0, 1, 0), 0.0);
}

TEST(TwoElectronBufferTest, LinearIndexComputation) {
    // Test that linear indexing matches expected formula:
    // index = a * (nb*nc*nd) + b * (nc*nd) + c * nd + d

    TwoElectronBuffer<0> buffer(2, 3, 4, 5);
    buffer.clear();

    const int nb = 3, nc = 4, nd = 5;

    // Set value at (1, 2, 3, 4)
    buffer(1, 2, 3, 4) = 42.0;

    // Compute expected linear index
    Size expected_index = 1 * (nb * nc * nd) + 2 * (nc * nd) + 3 * nd + 4;
    EXPECT_EQ(expected_index, 1 * 60 + 2 * 20 + 3 * 5 + 4);  // = 60 + 40 + 15 + 4 = 119

    // Verify it's at the correct position in the underlying data
    EXPECT_EQ(buffer.data()[expected_index], 42.0);
}

TEST(TwoElectronBufferTest, ConstAccessor) {
    TwoElectronBuffer<0> buffer(2, 2, 2, 2);
    buffer.clear();
    buffer(1, 1, 1, 1) = 3.14;

    const auto& const_buffer = buffer;
    EXPECT_EQ(const_buffer(1, 1, 1, 1), 3.14);
    EXPECT_EQ(const_buffer(0, 0, 0, 0), 0.0);
}

// ============================================================================
// Data Access
// ============================================================================

TEST(TwoElectronBufferTest, DataSpanReturnsCorrectSize) {
    TwoElectronBuffer<0> buffer(3, 3, 3, 3);

    auto span = buffer.data();
    EXPECT_EQ(span.size(), 81);

    auto const_span = std::as_const(buffer).data();
    EXPECT_EQ(const_span.size(), 81);
}

TEST(TwoElectronBufferTest, DataContiguity) {
    TwoElectronBuffer<0> buffer(2, 2, 2, 2);
    buffer.clear();

    // Set all values sequentially
    Real value = 1.0;
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < 2; ++c) {
                for (int d = 0; d < 2; ++d) {
                    buffer(a, b, c, d) = value++;
                }
            }
        }
    }

    // Verify data is contiguous and in expected order
    auto span = buffer.data();
    value = 1.0;
    int idx = 0;
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            for (int c = 0; c < 2; ++c) {
                for (int d = 0; d < 2; ++d) {
                    EXPECT_EQ(buffer(a, b, c, d), value) << "Mismatch at accessor (" << a << "," << b << "," << c << "," << d << ")";
                    EXPECT_EQ(span[idx], value) << "Mismatch at span index " << idx;
                    value++;
                    idx++;
                }
            }
        }
    }
}

TEST(TwoElectronBufferTest, DataPointerAccess) {
    TwoElectronBuffer<0> buffer(2, 2, 2, 2);
    buffer.clear();

    Real* ptr = buffer.data_ptr();
    const Real* const_ptr = std::as_const(buffer).data_ptr();

    EXPECT_NE(ptr, nullptr);
    EXPECT_NE(const_ptr, nullptr);
    EXPECT_EQ(ptr, const_ptr);

    // Modify via pointer
    ptr[0] = 123.456;
    EXPECT_EQ(buffer(0, 0, 0, 0), 123.456);
}

// ============================================================================
// Non-uniform Dimensions
// ============================================================================

TEST(TwoElectronBufferTest, NonUniformDimensions) {
    // Test (S|P) (P|D) shell quartet: na=1, nb=3, nc=3, nd=6
    TwoElectronBuffer<0> buffer(1, 3, 3, 6);

    EXPECT_EQ(buffer.na(), 1);
    EXPECT_EQ(buffer.nb(), 3);
    EXPECT_EQ(buffer.nc(), 3);
    EXPECT_EQ(buffer.nd(), 6);
    EXPECT_EQ(buffer.n_integrals(), 54);  // 1*3*3*6
    EXPECT_EQ(buffer.size(), 54);

    buffer.clear();

    // Set and verify a few values
    buffer(0, 0, 0, 0) = 1.0;
    buffer(0, 2, 2, 5) = 2.0;

    EXPECT_EQ(buffer(0, 0, 0, 0), 1.0);
    EXPECT_EQ(buffer(0, 2, 2, 5), 2.0);
}

// ============================================================================
// Type Alias
// ============================================================================

TEST(TwoElectronBufferTest, ERIBufferTypeAlias) {
    ERIBuffer buffer(3, 3, 3, 3);

    EXPECT_EQ(buffer.na(), 3);
    EXPECT_EQ(buffer.nb(), 3);
    EXPECT_EQ(buffer.nc(), 3);
    EXPECT_EQ(buffer.nd(), 3);
    EXPECT_EQ(buffer.size(), 81);

    buffer.clear();
    buffer(1, 1, 1, 1) = 2.718;
    EXPECT_EQ(buffer(1, 1, 1, 1), 2.718);
}

// ============================================================================
// Derivative Order Tests (DerivOrder=1)
// ============================================================================

TEST(TwoElectronBufferTest, GradientBufferNDerivComponents) {
    TwoElectronBuffer<1> buffer;

    // For 4 centers, DerivOrder=1: N_DERIV = 3 * 4 = 12
    EXPECT_EQ(buffer.N_DERIV, 12);
}

TEST(TwoElectronBufferTest, GradientBufferConstruction) {
    TwoElectronBuffer<1> buffer(2, 2, 2, 2);

    EXPECT_EQ(buffer.na(), 2);
    EXPECT_EQ(buffer.nb(), 2);
    EXPECT_EQ(buffer.nc(), 2);
    EXPECT_EQ(buffer.nd(), 2);
    EXPECT_EQ(buffer.n_integrals(), 16);   // 2*2*2*2
    EXPECT_EQ(buffer.size(), 192);         // 12 * 2*2*2*2
}

TEST(TwoElectronBufferTest, GradientBufferAccessor) {
    TwoElectronBuffer<1> buffer(2, 2, 2, 2);
    buffer.clear();

    // Set gradient components
    buffer(0, 0, 0, 0, 0) = 1.0;   // deriv component 0
    buffer(0, 0, 0, 0, 5) = 2.0;   // deriv component 5
    buffer(0, 0, 0, 0, 11) = 3.0;  // deriv component 11
    buffer(1, 1, 1, 1, 7) = 4.0;

    EXPECT_EQ(buffer(0, 0, 0, 0, 0), 1.0);
    EXPECT_EQ(buffer(0, 0, 0, 0, 5), 2.0);
    EXPECT_EQ(buffer(0, 0, 0, 0, 11), 3.0);
    EXPECT_EQ(buffer(1, 1, 1, 1, 7), 4.0);

    // Verify const accessor
    const auto& const_buffer = buffer;
    EXPECT_EQ(const_buffer(0, 0, 0, 0, 0), 1.0);
    EXPECT_EQ(const_buffer(1, 1, 1, 1, 7), 4.0);
}

TEST(TwoElectronBufferTest, GradientBufferClear) {
    TwoElectronBuffer<1> buffer(2, 2, 2, 2);

    // Fill with non-zero values
    for (int deriv = 0; deriv < 12; ++deriv) {
        buffer(0, 0, 0, 0, deriv) = static_cast<Real>(deriv + 1);
    }

    EXPECT_NE(buffer(0, 0, 0, 0, 5), 0.0);

    buffer.clear();

    for (int deriv = 0; deriv < 12; ++deriv) {
        EXPECT_EQ(buffer(0, 0, 0, 0, deriv), 0.0) << "Non-zero at deriv=" << deriv;
    }
}

// ============================================================================
// Derivative Order Tests (DerivOrder=2)
// ============================================================================

TEST(TwoElectronBufferTest, HessianBufferNDerivComponents) {
    TwoElectronBuffer<2> buffer;

    // For 4 centers, DerivOrder=2:
    // n_first = 3 * 4 = 12
    // N_DERIV = 12 * 13 / 2 = 78
    EXPECT_EQ(buffer.N_DERIV, 78);
}

TEST(TwoElectronBufferTest, HessianBufferConstruction) {
    TwoElectronBuffer<2> buffer(2, 2, 2, 2);

    EXPECT_EQ(buffer.na(), 2);
    EXPECT_EQ(buffer.nb(), 2);
    EXPECT_EQ(buffer.nc(), 2);
    EXPECT_EQ(buffer.nd(), 2);
    EXPECT_EQ(buffer.n_integrals(), 16);    // 2*2*2*2
    EXPECT_EQ(buffer.size(), 1248);         // 78 * 2*2*2*2
}

// ============================================================================
// Realistic Shell Quartet Sizes
// ============================================================================

TEST(TwoElectronBufferTest, SSSSShellQuartet) {
    // (S|S)(S|S): all na=nb=nc=nd=1
    TwoElectronBuffer<0> buffer(1, 1, 1, 1);

    EXPECT_EQ(buffer.n_integrals(), 1);
    EXPECT_EQ(buffer.size(), 1);

    buffer.clear();
    buffer(0, 0, 0, 0) = 0.12345;
    EXPECT_EQ(buffer(0, 0, 0, 0), 0.12345);
}

TEST(TwoElectronBufferTest, PPPPShellQuartet) {
    // (P|P)(P|P): all na=nb=nc=nd=3
    TwoElectronBuffer<0> buffer(3, 3, 3, 3);

    EXPECT_EQ(buffer.n_integrals(), 81);
    EXPECT_EQ(buffer.size(), 81);
}

TEST(TwoElectronBufferTest, DDDDShellQuartet) {
    // (D|D)(D|D): all na=nb=nc=nd=6
    TwoElectronBuffer<0> buffer(6, 6, 6, 6);

    EXPECT_EQ(buffer.n_integrals(), 1296);  // 6^4
    EXPECT_EQ(buffer.size(), 1296);
}

TEST(TwoElectronBufferTest, SPSDShellQuartet) {
    // (S|P)(S|D): na=1, nb=3, nc=1, nd=6
    TwoElectronBuffer<0> buffer(1, 3, 1, 6);

    EXPECT_EQ(buffer.n_integrals(), 18);  // 1*3*1*6
    EXPECT_EQ(buffer.size(), 18);

    buffer.clear();
    buffer(0, 2, 0, 5) = 99.0;
    EXPECT_EQ(buffer(0, 2, 0, 5), 99.0);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(TwoElectronBufferTest, SingleElementDimension) {
    TwoElectronBuffer<0> buffer(1, 10, 10, 10);

    EXPECT_EQ(buffer.na(), 1);
    EXPECT_EQ(buffer.n_integrals(), 1000);

    buffer.clear();
    buffer(0, 5, 5, 5) = 7.0;
    EXPECT_EQ(buffer(0, 5, 5, 5), 7.0);
}

TEST(TwoElectronBufferTest, LargeDimensions) {
    // Test with larger but realistic shell sizes (F shells)
    TwoElectronBuffer<0> buffer(10, 10, 10, 10);

    EXPECT_EQ(buffer.n_integrals(), 10000);
    EXPECT_EQ(buffer.size(), 10000);
}

// ============================================================================
// Edge Cases: Resize and Clear
// ============================================================================

TEST(TwoElectronBufferEdgeTest, ResizeToZero) {
    TwoElectronBuffer<0> buffer(2, 2, 2, 2);
    EXPECT_FALSE(buffer.empty());

    buffer.resize(0, 0, 0, 0);
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.na(), 0);
    EXPECT_EQ(buffer.nb(), 0);
    EXPECT_EQ(buffer.nc(), 0);
    EXPECT_EQ(buffer.nd(), 0);
}

TEST(TwoElectronBufferEdgeTest, ClearEmptyBuffer) {
    TwoElectronBuffer<0> buffer;
    EXPECT_TRUE(buffer.empty());
    // Should not crash
    buffer.clear();
    EXPECT_TRUE(buffer.empty());
}

// ============================================================================
// New Type Alias Tests
// ============================================================================

TEST(TwoElectronBufferTest, ERIGradientBufferAlias) {
    ERIGradientBuffer gbuf(2, 2, 2, 2);
    EXPECT_TRUE(gbuf.is_double_precision);
    EXPECT_EQ(gbuf.N_DERIV, 12);
    EXPECT_EQ(gbuf.size(), 192);
}

TEST(TwoElectronBufferTest, ERIHessianBufferAliases) {
    ERIHessianBuffer hbuf(2, 2, 2, 2);
    ERIHessianBufferFloat hbuf_f(2, 2, 2, 2);

    EXPECT_TRUE(hbuf.is_double_precision);
    EXPECT_EQ(hbuf.N_DERIV, 78);
    EXPECT_EQ(hbuf.size(), 1248);

    EXPECT_TRUE(hbuf_f.is_single_precision);
    EXPECT_EQ(hbuf_f.N_DERIV, 78);
    EXPECT_EQ(hbuf_f.size(), 1248);
}

}  // namespace libaccint
