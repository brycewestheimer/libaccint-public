// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file test_aligned_alloc.cpp
/// @brief Unit tests for aligned memory allocation utilities (Task 1.3.1)

#include <libaccint/utils/aligned_alloc.hpp>

#include <gtest/gtest.h>
#include <cstdint>
#include <numeric>
#include <vector>

using namespace libaccint::memory;

// ============================================================================
// aligned_malloc / aligned_free tests
// ============================================================================

TEST(AlignedAlloc, ReturnsAlignedPointer_16) {
    void* ptr = aligned_malloc(256, 16);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 16, 0u);
    aligned_free(ptr);
}

TEST(AlignedAlloc, ReturnsAlignedPointer_32) {
    void* ptr = aligned_malloc(256, 32);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 32, 0u);
    aligned_free(ptr);
}

TEST(AlignedAlloc, ReturnsAlignedPointer_64) {
    void* ptr = aligned_malloc(256, 64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 64, 0u);
    aligned_free(ptr);
}

TEST(AlignedAlloc, ReturnsAlignedPointer_128) {
    void* ptr = aligned_malloc(256, 128);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 128, 0u);
    aligned_free(ptr);
}

TEST(AlignedAlloc, ReturnsAlignedPointer_256) {
    void* ptr = aligned_malloc(1024, 256);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 256, 0u);
    aligned_free(ptr);
}

TEST(AlignedAlloc, ZeroSize) {
    // aligned_malloc(0) returns nullptr by design
    void* ptr = aligned_malloc(0, 64);
    EXPECT_EQ(ptr, nullptr);
    // aligned_free(nullptr) is safe
    aligned_free(ptr);
}

TEST(AlignedAlloc, LargeAllocation) {
    // 16 MB allocation
    constexpr std::size_t size = 16 * 1024 * 1024;
    void* ptr = aligned_malloc(size, 64);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 64, 0u);
    // Touch memory to ensure it's usable
    std::memset(ptr, 0, size);
    aligned_free(ptr);
}

TEST(AlignedAlloc, DeallocateDoesNotCrash) {
    for (int i = 0; i < 100; ++i) {
        void* ptr = aligned_malloc(128, 64);
        ASSERT_NE(ptr, nullptr);
        aligned_free(ptr);
    }
}

TEST(AlignedAlloc, MultipleAllocations) {
    std::vector<void*> ptrs;
    for (int i = 0; i < 50; ++i) {
        void* ptr = aligned_malloc(256, 64);
        ASSERT_NE(ptr, nullptr);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 64, 0u);
        ptrs.push_back(ptr);
    }
    // Deallocate in reverse order
    for (auto it = ptrs.rbegin(); it != ptrs.rend(); ++it) {
        aligned_free(*it);
    }
}

TEST(AlignedAlloc, FreeNullptrIsSafe) {
    aligned_free(nullptr);  // Should not crash
}

// ============================================================================
// AlignedBuffer tests
// ============================================================================

TEST(AlignedBuffer, DefaultConstructEmpty) {
    AlignedBuffer<double> buf;
    EXPECT_TRUE(buf.empty());
    EXPECT_EQ(buf.size(), 0u);
    EXPECT_EQ(buf.data(), nullptr);
}

TEST(AlignedBuffer, ConstructWithSize) {
    AlignedBuffer<double> buf(100);
    EXPECT_FALSE(buf.empty());
    EXPECT_EQ(buf.size(), 100u);
    ASSERT_NE(buf.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buf.data()) % 64, 0u);
}

TEST(AlignedBuffer, ConstructWithValue) {
    AlignedBuffer<double> buf(10, 3.14);
    EXPECT_EQ(buf.size(), 10u);
    for (std::size_t i = 0; i < buf.size(); ++i) {
        EXPECT_DOUBLE_EQ(buf[i], 3.14);
    }
}

TEST(AlignedBuffer, DataIsAligned) {
    for (std::size_t alignment : {16u, 32u, 64u}) {
        if (alignment == 16) {
            AlignedBuffer<double, 16> buf(128);
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buf.data()) % 16, 0u);
        } else if (alignment == 32) {
            AlignedBuffer<double, 32> buf(128);
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buf.data()) % 32, 0u);
        } else {
            AlignedBuffer<double, 64> buf(128);
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buf.data()) % 64, 0u);
        }
    }
}

TEST(AlignedBuffer, CopyConstruct) {
    AlignedBuffer<double> original(10, 2.5);
    AlignedBuffer<double> copy(original);
    EXPECT_EQ(copy.size(), original.size());
    for (std::size_t i = 0; i < copy.size(); ++i) {
        EXPECT_DOUBLE_EQ(copy[i], original[i]);
    }
}

TEST(AlignedBuffer, MoveConstruct) {
    AlignedBuffer<double> original(10, 2.5);
    auto* data_ptr = original.data();
    AlignedBuffer<double> moved(std::move(original));
    EXPECT_EQ(moved.data(), data_ptr);
    EXPECT_EQ(moved.size(), 10u);
    EXPECT_TRUE(original.empty());
}

TEST(AlignedBuffer, Resize) {
    AlignedBuffer<double> buf(10, 1.0);
    buf.resize(20);
    EXPECT_EQ(buf.size(), 20u);
    // Original elements should be preserved
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(buf[i], 1.0);
    }
}

TEST(AlignedBuffer, ZeroFill) {
    AlignedBuffer<double> buf(10, 99.0);
    buf.zero();
    for (std::size_t i = 0; i < buf.size(); ++i) {
        EXPECT_DOUBLE_EQ(buf[i], 0.0);
    }
}

TEST(AlignedBuffer, Iterators) {
    AlignedBuffer<double> buf(5);
    std::iota(buf.begin(), buf.end(), 0.0);
    for (std::size_t i = 0; i < buf.size(); ++i) {
        EXPECT_DOUBLE_EQ(buf[i], static_cast<double>(i));
    }
}

TEST(AlignedBuffer, AtBoundsCheck) {
    AlignedBuffer<double> buf(5, 1.0);
    EXPECT_NO_THROW(buf.at(4));
    EXPECT_THROW(buf.at(5), std::out_of_range);
}

// ============================================================================
// AlignedAllocator with STL containers
// ============================================================================

TEST(AlignedAllocator, AllocateAndDeallocate) {
    AlignedAllocator<double> alloc;
    double* ptr = alloc.allocate(100);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % DEFAULT_ALIGNMENT, 0u);
    // Write to memory to ensure it's usable
    for (int i = 0; i < 100; ++i) {
        ptr[i] = static_cast<double>(i);
    }
    alloc.deallocate(ptr, 100);
}

TEST(AlignedAllocator, AllocateZero) {
    AlignedAllocator<double> alloc;
    double* ptr = alloc.allocate(0);
    EXPECT_EQ(ptr, nullptr);
}

TEST(AlignedAllocator, Equality) {
    AlignedAllocator<double> a;
    AlignedAllocator<double> b;
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
}

// ============================================================================
// Utility functions
// ============================================================================

TEST(AlignedAlloc, IsAlignedCheck) {
    AlignedBuffer<double> buf(64);
    EXPECT_TRUE(is_cache_aligned(buf.data()));
    EXPECT_TRUE(is_simd_aligned(buf.data()));
}

TEST(AlignedAlloc, MakeAlignedBuffer) {
    auto buf = make_aligned_buffer<double>(20);
    EXPECT_EQ(buf.size(), 20u);
    EXPECT_TRUE(is_cache_aligned(buf.data()));
}

TEST(AlignedAlloc, MakeAlignedBufferWithValue) {
    auto buf = make_aligned_buffer<double>(10, 7.7);
    EXPECT_EQ(buf.size(), 10u);
    for (std::size_t i = 0; i < buf.size(); ++i) {
        EXPECT_DOUBLE_EQ(buf[i], 7.7);
    }
}
