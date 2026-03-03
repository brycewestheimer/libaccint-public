// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file aligned_alloc.hpp
/// @brief Cache-aligned memory allocation utilities for SIMD operations
///
/// Provides RAII wrappers for aligned memory allocation to ensure proper
/// alignment for SIMD operations and cache-friendly memory access.

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
    #include <malloc.h>
#endif

namespace libaccint::memory {

// ============================================================================
// Alignment Constants
// ============================================================================

/// @brief Default alignment for cache-line optimization (64 bytes)
///
/// 64 bytes is the most common cache line size on modern x86 processors.
/// This alignment prevents false sharing and enables efficient SIMD operations.
inline constexpr std::size_t DEFAULT_ALIGNMENT = 64;

/// @brief SIMD alignment requirement (AVX2 = 32 bytes, AVX-512 = 64 bytes)
inline constexpr std::size_t SIMD_ALIGNMENT = 64;

// ============================================================================
// Aligned Allocation Functions
// ============================================================================

/// @brief Allocate aligned memory
/// @param size Number of bytes to allocate
/// @param alignment Alignment requirement (must be power of 2)
/// @return Pointer to aligned memory, or nullptr on failure
/// @note Caller must use aligned_free() to deallocate
inline void* aligned_malloc(std::size_t size, std::size_t alignment = DEFAULT_ALIGNMENT) {
    if (size == 0) {
        return nullptr;
    }

#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    // C11 aligned_alloc requires size to be multiple of alignment
    std::size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    return std::aligned_alloc(alignment, aligned_size);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

/// @brief Free aligned memory
/// @param ptr Pointer returned by aligned_malloc
inline void aligned_free(void* ptr) noexcept {
    if (ptr) {
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
}

// ============================================================================
// AlignedDeleter - Custom deleter for smart pointers
// ============================================================================

/// @brief Custom deleter for aligned memory
struct AlignedDeleter {
    void operator()(void* ptr) const noexcept {
        aligned_free(ptr);
    }
};

/// @brief Typed aligned deleter
template<typename T>
struct TypedAlignedDeleter {
    void operator()(T* ptr) const noexcept {
        if (ptr) {
            // Call destructor if not trivially destructible
            if constexpr (!std::is_trivially_destructible_v<T>) {
                ptr->~T();
            }
            aligned_free(ptr);
        }
    }
};

// ============================================================================
// AlignedBuffer - RAII wrapper for aligned arrays
// ============================================================================

/// @brief RAII wrapper for cache-aligned contiguous memory
///
/// Provides automatic memory management with guaranteed alignment.
/// Suitable for SIMD operations and cache-efficient data access.
///
/// @tparam T Element type
/// @tparam Alignment Memory alignment (default: 64 bytes for cache line)
template<typename T, std::size_t Alignment = DEFAULT_ALIGNMENT>
class AlignedBuffer {
public:
    static_assert(Alignment >= alignof(T),
                  "Alignment must be at least alignof(T)");
    static_assert((Alignment & (Alignment - 1)) == 0,
                  "Alignment must be a power of 2");

    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;

    static constexpr std::size_t alignment = Alignment;

    /// @brief Default constructor (empty buffer)
    AlignedBuffer() noexcept = default;

    /// @brief Construct buffer with specified size
    /// @param n Number of elements
    /// @throws std::bad_alloc if allocation fails
    explicit AlignedBuffer(size_type n)
        : size_(n)
        , capacity_(compute_padded_size(n))
    {
        if (n > 0) {
            data_ = static_cast<T*>(aligned_malloc(capacity_ * sizeof(T), Alignment));
            if (!data_) {
                throw std::bad_alloc();
            }
            // Default-construct elements if needed
            if constexpr (!std::is_trivially_default_constructible_v<T>) {
                for (size_type i = 0; i < size_; ++i) {
                    new (data_ + i) T();
                }
            }
        }
    }

    /// @brief Construct buffer with specified size and initial value
    /// @param n Number of elements
    /// @param value Initial value for all elements
    AlignedBuffer(size_type n, const T& value)
        : size_(n)
        , capacity_(compute_padded_size(n))
    {
        if (n > 0) {
            data_ = static_cast<T*>(aligned_malloc(capacity_ * sizeof(T), Alignment));
            if (!data_) {
                throw std::bad_alloc();
            }
            for (size_type i = 0; i < size_; ++i) {
                new (data_ + i) T(value);
            }
        }
    }

    /// @brief Destructor
    ~AlignedBuffer() {
        clear_and_free();
    }

    /// @brief Copy constructor
    AlignedBuffer(const AlignedBuffer& other)
        : size_(other.size_)
        , capacity_(other.capacity_)
    {
        if (other.data_) {
            data_ = static_cast<T*>(aligned_malloc(capacity_ * sizeof(T), Alignment));
            if (!data_) {
                throw std::bad_alloc();
            }
            for (size_type i = 0; i < size_; ++i) {
                new (data_ + i) T(other.data_[i]);
            }
        }
    }

    /// @brief Move constructor
    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(other.data_)
        , size_(other.size_)
        , capacity_(other.capacity_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    /// @brief Copy assignment
    AlignedBuffer& operator=(const AlignedBuffer& other) {
        if (this != &other) {
            AlignedBuffer tmp(other);
            swap(tmp);
        }
        return *this;
    }

    /// @brief Move assignment
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            clear_and_free();
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    /// @brief Swap contents with another buffer
    void swap(AlignedBuffer& other) noexcept {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }

    // =========================================================================
    // Element Access
    // =========================================================================

    /// @brief Access element at index
    [[nodiscard]] reference operator[](size_type i) noexcept {
        return data_[i];
    }

    /// @brief Access element at index (const)
    [[nodiscard]] const_reference operator[](size_type i) const noexcept {
        return data_[i];
    }

    /// @brief Access element with bounds checking
    [[nodiscard]] reference at(size_type i) {
        if (i >= size_) {
            throw std::out_of_range("AlignedBuffer::at: index out of range");
        }
        return data_[i];
    }

    /// @brief Access element with bounds checking (const)
    [[nodiscard]] const_reference at(size_type i) const {
        if (i >= size_) {
            throw std::out_of_range("AlignedBuffer::at: index out of range");
        }
        return data_[i];
    }

    /// @brief Get pointer to underlying data
    [[nodiscard]] pointer data() noexcept { return data_; }

    /// @brief Get pointer to underlying data (const)
    [[nodiscard]] const_pointer data() const noexcept { return data_; }

    /// @brief Access first element
    [[nodiscard]] reference front() noexcept { return data_[0]; }

    /// @brief Access first element (const)
    [[nodiscard]] const_reference front() const noexcept { return data_[0]; }

    /// @brief Access last element
    [[nodiscard]] reference back() noexcept { return data_[size_ - 1]; }

    /// @brief Access last element (const)
    [[nodiscard]] const_reference back() const noexcept { return data_[size_ - 1]; }

    // =========================================================================
    // Iterators
    // =========================================================================

    [[nodiscard]] iterator begin() noexcept { return data_; }
    [[nodiscard]] const_iterator begin() const noexcept { return data_; }
    [[nodiscard]] const_iterator cbegin() const noexcept { return data_; }

    [[nodiscard]] iterator end() noexcept { return data_ + size_; }
    [[nodiscard]] const_iterator end() const noexcept { return data_ + size_; }
    [[nodiscard]] const_iterator cend() const noexcept { return data_ + size_; }

    // =========================================================================
    // Capacity
    // =========================================================================

    /// @brief Get number of elements
    [[nodiscard]] size_type size() const noexcept { return size_; }

    /// @brief Get allocated capacity (may be larger than size due to padding)
    [[nodiscard]] size_type capacity() const noexcept { return capacity_; }

    /// @brief Check if buffer is empty
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// @brief Get size in bytes
    [[nodiscard]] size_type size_bytes() const noexcept { return size_ * sizeof(T); }

    // =========================================================================
    // Modifiers
    // =========================================================================

    /// @brief Resize buffer
    /// @param n New size
    ///
    /// If n > capacity, reallocates. New elements are default-constructed.
    void resize(size_type n) {
        if (n > capacity_) {
            // Need reallocation
            size_type new_capacity = compute_padded_size(n);
            T* new_data = static_cast<T*>(aligned_malloc(new_capacity * sizeof(T), Alignment));
            if (!new_data) {
                throw std::bad_alloc();
            }

            // Move existing elements
            for (size_type i = 0; i < size_; ++i) {
                new (new_data + i) T(std::move(data_[i]));
                if constexpr (!std::is_trivially_destructible_v<T>) {
                    data_[i].~T();
                }
            }

            // Default-construct new elements
            if constexpr (!std::is_trivially_default_constructible_v<T>) {
                for (size_type i = size_; i < n; ++i) {
                    new (new_data + i) T();
                }
            }

            aligned_free(data_);
            data_ = new_data;
            capacity_ = new_capacity;
        } else if (n > size_) {
            // Default-construct new elements
            if constexpr (!std::is_trivially_default_constructible_v<T>) {
                for (size_type i = size_; i < n; ++i) {
                    new (data_ + i) T();
                }
            }
        } else if (n < size_) {
            // Destroy excess elements
            if constexpr (!std::is_trivially_destructible_v<T>) {
                for (size_type i = n; i < size_; ++i) {
                    data_[i].~T();
                }
            }
        }
        size_ = n;
    }

    /// @brief Resize buffer and fill with value
    void resize(size_type n, const T& value) {
        size_type old_size = size_;
        resize(n);
        for (size_type i = old_size; i < n; ++i) {
            data_[i] = value;
        }
    }

    /// @brief Clear contents (does not deallocate)
    void clear() noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (size_type i = 0; i < size_; ++i) {
                data_[i].~T();
            }
        }
        size_ = 0;
    }

    /// @brief Fill all elements with zero (for numeric types)
    void zero() noexcept requires std::is_arithmetic_v<T> {
        std::memset(data_, 0, capacity_ * sizeof(T));
    }

    /// @brief Fill all elements with value
    void fill(const T& value) noexcept {
        for (size_type i = 0; i < size_; ++i) {
            data_[i] = value;
        }
    }

private:
    T* data_{nullptr};
    size_type size_{0};
    size_type capacity_{0};

    /// @brief Compute padded size to be multiple of SIMD width
    static constexpr size_type compute_padded_size(size_type n) noexcept {
        // Pad to multiple of 8 doubles (64 bytes / sizeof(double))
        constexpr size_type simd_elements = SIMD_ALIGNMENT / sizeof(T);
        return ((n + simd_elements - 1) / simd_elements) * simd_elements;
    }

    /// @brief Clear and free memory
    void clear_and_free() noexcept {
        if (data_) {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                for (size_type i = 0; i < size_; ++i) {
                    data_[i].~T();
                }
            }
            aligned_free(data_);
            data_ = nullptr;
        }
        size_ = 0;
        capacity_ = 0;
    }
};

// ============================================================================
// AlignedAllocator - STL-compatible allocator
// ============================================================================

/// @brief STL-compatible allocator for aligned memory
///
/// Can be used with std::vector and other STL containers to ensure
/// aligned memory allocation.
///
/// @tparam T Element type
/// @tparam Alignment Memory alignment
template<typename T, std::size_t Alignment = DEFAULT_ALIGNMENT>
class AlignedAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    static constexpr std::size_t alignment = Alignment;

    AlignedAllocator() noexcept = default;

    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(size_type n) {
        if (n == 0) {
            return nullptr;
        }

        // Ensure allocation size is multiple of alignment
        size_type bytes = n * sizeof(T);
        size_type aligned_bytes = ((bytes + Alignment - 1) / Alignment) * Alignment;

        void* ptr = aligned_malloc(aligned_bytes, Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }

        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, [[maybe_unused]] size_type n) noexcept {
        aligned_free(ptr);
    }

    template<typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
        return true;
    }

    template<typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const noexcept {
        return false;
    }
};

// ============================================================================
// Type Aliases
// ============================================================================

/// @brief Aligned vector of doubles (64-byte alignment)
using AlignedDoubleBuffer = AlignedBuffer<double, 64>;

/// @brief Aligned vector of floats (64-byte alignment)
using AlignedFloatBuffer = AlignedBuffer<float, 64>;

/// @brief STL vector with aligned allocation
template<typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, DEFAULT_ALIGNMENT>>;

// ============================================================================
// Utility Functions
// ============================================================================

/// @brief Create an aligned buffer with specified size
template<typename T, std::size_t Alignment = DEFAULT_ALIGNMENT>
[[nodiscard]] AlignedBuffer<T, Alignment> make_aligned_buffer(std::size_t n) {
    return AlignedBuffer<T, Alignment>(n);
}

/// @brief Create an aligned buffer with specified size and initial value
template<typename T, std::size_t Alignment = DEFAULT_ALIGNMENT>
[[nodiscard]] AlignedBuffer<T, Alignment> make_aligned_buffer(std::size_t n, const T& value) {
    return AlignedBuffer<T, Alignment>(n, value);
}

/// @brief Check if pointer is properly aligned
template<std::size_t Alignment>
[[nodiscard]] inline bool is_aligned(const void* ptr) noexcept {
    return reinterpret_cast<std::uintptr_t>(ptr) % Alignment == 0;
}

/// @brief Check if pointer is cache-line aligned (64 bytes)
[[nodiscard]] inline bool is_cache_aligned(const void* ptr) noexcept {
    return is_aligned<DEFAULT_ALIGNMENT>(ptr);
}

/// @brief Check if pointer is SIMD aligned
[[nodiscard]] inline bool is_simd_aligned(const void* ptr) noexcept {
    return is_aligned<SIMD_ALIGNMENT>(ptr);
}

}  // namespace libaccint::memory
