// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file memory_pool.hpp
/// @brief Memory pooling for efficient scratch buffer allocation
///
/// Provides thread-local memory pools to reduce allocation overhead in
/// hot paths. Each thread has its own pool to avoid synchronization.

#include <libaccint/utils/aligned_alloc.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace libaccint::memory {

// ============================================================================
// Pool Configuration
// ============================================================================

/// @brief Size classes for pooled allocations
///
/// Allocations are rounded up to the nearest size class.
/// Size classes are chosen to cover common integral computation buffers.
namespace pool_config {

/// @brief Number of size classes
inline constexpr std::size_t NUM_SIZE_CLASSES = 8;

/// @brief Size class boundaries (in bytes)
///
/// Class 0: up to 256 bytes
/// Class 1: up to 1024 bytes
/// Class 2: up to 4096 bytes
/// Class 3: up to 16384 bytes
/// Class 4: up to 65536 bytes
/// Class 5: up to 262144 bytes
/// Class 6: up to 1048576 bytes
/// Class 7: up to 4194304 bytes (4 MB)
inline constexpr std::array<std::size_t, NUM_SIZE_CLASSES> SIZE_CLASSES = {
    256,        // 256 B - small scratch (indices, counters)
    1024,       // 1 KB - small recursion tables
    4096,       // 4 KB - medium recursion tables
    16384,      // 16 KB - large recursion tables
    65536,      // 64 KB - integral buffers
    262144,     // 256 KB - batch buffers
    1048576,    // 1 MB - large batch buffers
    4194304     // 4 MB - maximum pooled size
};

/// @brief Maximum number of buffers to keep per size class
inline constexpr std::size_t MAX_BUFFERS_PER_CLASS = 8;

/// @brief Find the size class for a given allocation size
/// @return Size class index, or NUM_SIZE_CLASSES if too large
inline constexpr std::size_t find_size_class(std::size_t bytes) noexcept {
    for (std::size_t i = 0; i < NUM_SIZE_CLASSES; ++i) {
        if (bytes <= SIZE_CLASSES[i]) {
            return i;
        }
    }
    return NUM_SIZE_CLASSES;  // Too large for pooling
}

}  // namespace pool_config

// ============================================================================
// PooledBuffer - RAII wrapper for pooled memory
// ============================================================================

class MemoryPool;  // Forward declaration

/// @brief RAII guard for acquiring and releasing pooled memory
///
/// Automatically returns memory to the pool on destruction.
/// Thread-safe when used with the thread-local pool.
class PooledBuffer {
public:
    /// @brief Default constructor (empty buffer)
    PooledBuffer() noexcept = default;

    /// @brief Construct with allocated memory
    /// @param ptr Pointer to pooled memory
    /// @param size Allocated size in bytes
    /// @param pool Pool to return memory to (may be nullptr for non-pooled)
    /// @param size_class Size class index
    PooledBuffer(void* ptr, std::size_t size, MemoryPool* pool, std::size_t size_class) noexcept
        : ptr_(ptr)
        , size_(size)
        , pool_(pool)
        , size_class_(size_class)
    {}

    /// @brief Destructor - returns memory to pool
    ~PooledBuffer();

    /// @brief No copy
    PooledBuffer(const PooledBuffer&) = delete;
    PooledBuffer& operator=(const PooledBuffer&) = delete;

    /// @brief Move constructor
    PooledBuffer(PooledBuffer&& other) noexcept
        : ptr_(other.ptr_)
        , size_(other.size_)
        , pool_(other.pool_)
        , size_class_(other.size_class_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.pool_ = nullptr;
        other.size_class_ = 0;
    }

    /// @brief Move assignment
    PooledBuffer& operator=(PooledBuffer&& other) noexcept;

    /// @brief Get pointer to allocated memory
    [[nodiscard]] void* data() noexcept { return ptr_; }
    [[nodiscard]] const void* data() const noexcept { return ptr_; }

    /// @brief Get typed pointer
    template<typename T>
    [[nodiscard]] T* as() noexcept {
        return static_cast<T*>(ptr_);
    }

    template<typename T>
    [[nodiscard]] const T* as() const noexcept {
        return static_cast<const T*>(ptr_);
    }

    /// @brief Get allocated size in bytes
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    /// @brief Check if buffer is valid
    [[nodiscard]] explicit operator bool() const noexcept { return ptr_ != nullptr; }

    /// @brief Release ownership and return pointer (does NOT return to pool)
    [[nodiscard]] void* release() noexcept {
        void* p = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        pool_ = nullptr;
        return p;
    }

private:
    void* ptr_{nullptr};
    std::size_t size_{0};
    MemoryPool* pool_{nullptr};
    std::size_t size_class_{0};
};

// ============================================================================
// MemoryPool - Per-thread memory pool
// ============================================================================

/// @brief Thread-local memory pool for efficient scratch buffer allocation
///
/// Maintains free lists for each size class. Memory is returned to the
/// appropriate free list on release. When the free list is empty, a new
/// allocation is made. When the free list is full, memory is freed.
///
/// All allocations are 64-byte aligned for cache and SIMD efficiency.
class MemoryPool {
public:
    /// @brief Constructor
    MemoryPool() = default;

    /// @brief Destructor - frees all pooled memory
    virtual ~MemoryPool() {
        clear();
    }

    /// @brief No copy
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    /// @brief Acquire a buffer of at least the specified size
    /// @param bytes Minimum size in bytes
    /// @return PooledBuffer with allocated memory
    [[nodiscard]] PooledBuffer acquire(std::size_t bytes);

    /// @brief Acquire typed buffer
    /// @tparam T Element type
    /// @param count Number of elements
    /// @return PooledBuffer with allocated memory
    template<typename T>
    [[nodiscard]] PooledBuffer acquire_typed(std::size_t count) {
        return acquire(count * sizeof(T));
    }

    /// @brief Release memory back to the pool
    /// @param ptr Pointer to memory
    /// @param size_class Size class of the allocation
    ///
    /// Called automatically by PooledBuffer destructor.
    /// Virtual so GlobalMemoryPool can override with mutex-protected version.
    virtual void release(void* ptr, std::size_t size_class) noexcept;

    /// @brief Free all pooled memory
    void clear() noexcept;

    /// @brief Trim each free list to at most max_per_class entries
    /// @param max_per_class Maximum buffers to retain per size class
    void trim(std::size_t max_per_class = pool_config::MAX_BUFFERS_PER_CLASS / 2) noexcept;

    /// @brief Get statistics about pool usage
    struct Stats {
        std::size_t total_allocations{0};      ///< Total acquire() calls
        std::size_t pool_hits{0};              ///< Reused from pool
        std::size_t pool_misses{0};            ///< New allocations
        std::size_t current_pooled{0};         ///< Currently pooled buffers
        std::size_t current_pooled_bytes{0};   ///< Currently pooled bytes
        std::size_t oversized_allocations{0};  ///< Allocations too large for pool
        std::size_t peak_pooled_bytes{0};      ///< High-water mark of pooled bytes
    };

    [[nodiscard]] Stats stats() const noexcept;

private:
    /// @brief Free list for each size class
    struct FreeList {
        std::vector<void*> buffers;
        std::size_t buffer_size{0};
    };

    std::array<FreeList, pool_config::NUM_SIZE_CLASSES> free_lists_;

    // Statistics
    std::size_t total_allocations_{0};
    std::size_t pool_hits_{0};
    std::size_t pool_misses_{0};
    std::size_t oversized_allocations_{0};
    std::size_t peak_pooled_bytes_{0};
};

// ============================================================================
// Thread-Local Pool Access
// ============================================================================

/// @brief Get the thread-local memory pool
///
/// Each thread has its own pool instance, avoiding synchronization overhead.
/// The pool is lazily initialized on first access.
[[nodiscard]] MemoryPool& get_thread_local_pool() noexcept;

/// @brief Acquire memory from the thread-local pool
/// @param bytes Minimum size in bytes
/// @return PooledBuffer with allocated memory
[[nodiscard]] inline PooledBuffer pool_acquire(std::size_t bytes) {
    return get_thread_local_pool().acquire(bytes);
}

/// @brief Acquire typed memory from the thread-local pool
/// @tparam T Element type
/// @param count Number of elements
/// @return PooledBuffer with allocated memory
template<typename T>
[[nodiscard]] inline PooledBuffer pool_acquire_typed(std::size_t count) {
    return get_thread_local_pool().acquire_typed<T>(count);
}

/// @brief Clear the thread-local pool, freeing all cached memory
///
/// Useful when transitioning between computation phases to reclaim memory.
inline void pool_clear() noexcept {
    get_thread_local_pool().clear();
}

/// @brief Trim the thread-local pool, reducing each free list to at most max_per_class entries
/// @param max_per_class Maximum buffers to retain per size class
inline void pool_trim(std::size_t max_per_class = pool_config::MAX_BUFFERS_PER_CLASS / 2) noexcept {
    get_thread_local_pool().trim(max_per_class);
}

// ============================================================================
// Global Pool for Shared Resources
// ============================================================================

/// @brief Thread-safe global memory pool for shared resources
///
/// Unlike the thread-local pool, this pool uses locking and should only
/// be used for resources that must be shared across threads.
///
/// Inherits from MemoryPool so that PooledBuffer objects created by
/// acquire() have their pool_ back-pointer set to this GlobalMemoryPool.
/// Since release() is virtual, PooledBuffer destructors will dispatch
/// to the mutex-protected GlobalMemoryPool::release() override.
class GlobalMemoryPool : public MemoryPool {
public:
    /// @brief Get the singleton instance
    [[nodiscard]] static GlobalMemoryPool& instance();

    /// @brief Acquire a buffer (thread-safe)
    [[nodiscard]] PooledBuffer acquire(std::size_t bytes);

    /// @brief Release memory back to the pool (thread-safe)
    /// Overrides MemoryPool::release() to add mutex protection.
    void release(void* ptr, std::size_t size_class) noexcept override;

    /// @brief Clear all pooled memory (thread-safe)
    void clear() noexcept;

    /// @brief Get pool statistics (thread-safe, consistent snapshot)
    [[nodiscard]] Stats stats() const noexcept;

private:
    GlobalMemoryPool() = default;
    ~GlobalMemoryPool() override;

    GlobalMemoryPool(const GlobalMemoryPool&) = delete;
    GlobalMemoryPool& operator=(const GlobalMemoryPool&) = delete;

    mutable std::mutex mutex_;
};

// ============================================================================
// ScopedPoolClear - RAII pool cleanup
// ============================================================================

/// @brief RAII guard that clears the thread-local pool on destruction
///
/// Use at the end of a computation phase to ensure pooled memory is released.
class ScopedPoolClear {
public:
    ScopedPoolClear() = default;

    ~ScopedPoolClear() {
        pool_clear();
    }

    ScopedPoolClear(const ScopedPoolClear&) = delete;
    ScopedPoolClear& operator=(const ScopedPoolClear&) = delete;
};

}  // namespace libaccint::memory
